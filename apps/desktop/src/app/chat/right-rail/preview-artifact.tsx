import { useStore } from '@nanostores/react'
import DOMPurify from 'dompurify'
import { useEffect, useMemo, useState } from 'react'

import { CopyButton } from '@/components/ui/copy-button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { artifactDownloadName, type ArtifactKind } from '@/lib/artifact-detect'
import { downloadTextFile } from '@/lib/download-text'
import { ChevronLeft, ChevronRight, Download, ExternalLink } from '@/lib/icons'
import { $artifactRegistry, $artifactVersionSelection, findArtifact, selectArtifactVersion } from '@/store/artifacts'
import { notifyError } from '@/store/notifications'
import type { PreviewTarget } from '@/store/preview'

import { PreviewEmptyState, PreviewModeSwitcher, type PreviewViewMode, SourceView } from './preview-file'

const MIME_BY_KIND = { code: 'text/plain', html: 'text/html', svg: 'image/svg+xml' } as const

// Shiki has no `svg` grammar; code artifacts keep their detected fence language.
const SOURCE_LANGUAGE_BY_KIND: Record<ArtifactKind, string | undefined> = {
  code: undefined,
  html: 'html',
  svg: 'xml'
}

const HEADER_BUTTON_CLASS =
  'flex items-center gap-1 rounded-md px-1.5 text-[0.625rem] font-bold text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-40'

/** Wrap an HTML fragment in a minimal document shell; full documents pass
 *  through untouched. Keeps generated fragments (no <html>/<body>) rendering
 *  with sane defaults instead of quirks-mode soup. */
function composeArtifactHtml(content: string): string {
  if (/<html[\s>]|<!doctype\s+html/i.test(content)) {
    return content
  }

  return [
    '<!doctype html>',
    '<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
    '<style>body{margin:0;font-family:system-ui,sans-serif}</style></head><body>',
    content,
    '</body></html>'
  ].join('\n')
}

/** Write the composed document to a real temp file through the existing
 *  buffer-save IPC, then hand it to the OS browser. A blob/data URL can't
 *  cross into the OS default browser, so a file on disk is the honest path. */
async function openHtmlInBrowser(content: string): Promise<void> {
  const bridge = window.hermesDesktop

  if (!bridge?.saveImageBuffer || !bridge.openExternal) {
    throw new Error('Desktop bridge unavailable')
  }

  const bytes = new TextEncoder().encode(composeArtifactHtml(content))
  const path = await bridge.saveImageBuffer(bytes, '.html')

  if (!path) {
    throw new Error('Could not write artifact file')
  }

  const fileUrl = `file://${path.startsWith('/') ? '' : '/'}${path.replace(/\\/g, '/')}`

  if (bridge.openPreviewInBrowser) {
    await bridge.openPreviewInBrowser(fileUrl)

    return
  }

  await bridge.openExternal(fileUrl)
}

/**
 * Live view for renderable artifact content.
 *
 * HTML runs in an `<iframe sandbox="allow-scripts">` — scripts execute in an
 * opaque origin with no same-origin access, no top navigation, no popups, no
 * form submission out of the frame. The parent app is unreachable. SVG is
 * DOMPurify-sanitized with the same profile as the inline ```svg embed.
 */
function ArtifactLiveView({ content, kind, title }: { content: string; kind: ArtifactKind; title: string }) {
  const svgClean = useMemo(
    () => (kind === 'svg' ? DOMPurify.sanitize(content, { USE_PROFILES: { svg: true, svgFilters: true } }) : ''),
    [content, kind]
  )

  if (kind === 'svg') {
    return (
      <div className="grid h-full place-items-center overflow-auto bg-background p-4 [&_svg]:h-auto [&_svg]:max-h-full [&_svg]:w-auto [&_svg]:max-w-full">
        <div dangerouslySetInnerHTML={{ __html: svgClean }} />
      </div>
    )
  }

  return (
    <iframe
      className="block size-full border-0 bg-white"
      sandbox="allow-scripts"
      srcDoc={composeArtifactHtml(content)}
      // Deliberately raw white + forced light scheme: the frame hosts foreign
      // generated HTML that assumes a light canvas, so it renders deterministically
      // light in both app themes instead of inheriting theme tokens it can't see.
      style={{ colorScheme: 'light' }}
      title={title}
    />
  )
}

function VersionStepper({
  current,
  onSelect,
  total
}: {
  current: number
  onSelect: (index: number) => void
  total: number
}) {
  const { t } = useI18n()
  const copy = t.artifactPreview

  if (total < 2) {
    return null
  }

  return (
    <>
      <Tip label={copy.olderVersion}>
        <button
          aria-label={copy.olderVersion}
          className={HEADER_BUTTON_CLASS}
          disabled={current === 0}
          onClick={() => onSelect(current - 1)}
          type="button"
        >
          <ChevronLeft className="size-3" />
        </button>
      </Tip>
      <span className="text-[0.625rem] font-bold tabular-nums text-muted-foreground">
        {copy.versionOf(current + 1, total)}
      </span>
      <Tip label={copy.newerVersion}>
        <button
          aria-label={copy.newerVersion}
          className={HEADER_BUTTON_CLASS}
          disabled={current === total - 1}
          onClick={() => onSelect(current + 1)}
          type="button"
        >
          <ChevronRight className="size-3" />
        </button>
      </Tip>
      {current < total - 1 && (
        <button
          className="text-[0.625rem] font-bold text-muted-foreground underline decoration-current/25 underline-offset-4 transition-colors hover:text-foreground"
          onClick={() => onSelect(total - 1)}
          type="button"
        >
          {copy.latest}
        </button>
      )}
    </>
  )
}

/**
 * Renders an `artifact` preview target inside the real preview pane: the
 * shared mode switcher up top (RENDERED / SOURCE, plus the version stepper and
 * content actions in its trailing slot) over either the live view or the same
 * windowed source view a file preview uses.
 *
 * The target only carries the artifact id — content is read live from the
 * registry, so an open tab picks up new versions as the model iterates.
 */
export function ArtifactPreview({ target }: { target: PreviewTarget }) {
  const { t } = useI18n()
  const copy = t.artifactPreview
  const artifactId = target.url
  const registry = useStore($artifactRegistry)
  const versionSelection = useStore($artifactVersionSelection)
  const [userMode, setUserMode] = useState<PreviewViewMode | null>(null)

  // Reset the explicit mode when the pane is reused for another artifact.
  useEffect(() => {
    setUserMode(null)
  }, [artifactId])

  const record = useMemo(() => findArtifact(registry, artifactId), [artifactId, registry])

  if (!record) {
    return <PreviewEmptyState body={copy.missingBody} title={copy.missingTitle} />
  }

  const renderable = record.kind === 'html' || record.kind === 'svg'
  const modes: PreviewViewMode[] = renderable ? ['rendered', 'source'] : ['source']
  const mode = userMode && modes.includes(userMode) ? userMode : modes[0]!
  const versionIndex = Math.min(versionSelection[artifactId] ?? record.versions.length - 1, record.versions.length - 1)
  const version = record.versions[versionIndex]!

  return (
    <div className="flex h-full flex-col overflow-hidden bg-transparent">
      <PreviewModeSwitcher
        active={mode}
        modes={modes}
        onSelect={setUserMode}
        trailing={
          <>
            <VersionStepper
              current={versionIndex}
              onSelect={index => selectArtifactVersion(artifactId, index)}
              total={record.versions.length}
            />
            <CopyButton
              appearance="inline"
              className="h-5 px-1 opacity-70 hover:opacity-100"
              iconClassName="size-3"
              label={copy.copyContent}
              showLabel={false}
              text={version.content}
            />
            <Tip label={copy.download}>
              <button
                aria-label={copy.download}
                className={HEADER_BUTTON_CLASS}
                onClick={() =>
                  downloadTextFile(
                    artifactDownloadName(record.kind, record.language, record.title),
                    version.content,
                    MIME_BY_KIND[record.kind]
                  )
                }
                type="button"
              >
                <Download className="size-3" />
              </button>
            </Tip>
            {record.kind === 'html' && window.hermesDesktop && (
              <Tip label={copy.openInBrowser}>
                <button
                  aria-label={copy.openInBrowser}
                  className={HEADER_BUTTON_CLASS}
                  onClick={() =>
                    void openHtmlInBrowser(version.content).catch(error => notifyError(error, copy.openInBrowserFailed))
                  }
                  type="button"
                >
                  <ExternalLink className="size-3" />
                </button>
              </Tip>
            )}
          </>
        }
      />
      <div className="min-h-0 flex-1 overflow-hidden">
        {mode === 'rendered' && renderable ? (
          <ArtifactLiveView content={version.content} kind={record.kind} title={record.title} />
        ) : (
          <SourceView language={SOURCE_LANGUAGE_BY_KIND[record.kind] ?? record.language} text={version.content} />
        )}
      </div>
    </div>
  )
}
