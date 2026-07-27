import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { CopyButton } from '@/components/ui/copy-button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { artifactDownloadName } from '@/lib/artifact-detect'
import { downloadTextFile } from '@/lib/download-text'
import { ChevronLeft, ChevronRight, Download, ExternalLink } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $artifactRegistry,
  $artifactVersionSelection,
  type ArtifactRecord,
  selectArtifactVersion
} from '@/store/artifacts'
import { notifyError } from '@/store/notifications'

import { ArtifactLivePreview, ArtifactSourceView, composeArtifactHtml } from './artifact-renderers'
import { PreviewEmptyState } from './preview-file'

type ArtifactViewMode = 'preview' | 'source'

const MIME_BY_KIND = { code: 'text/plain', html: 'text/html', svg: 'image/svg+xml' } as const

const HEADER_BUTTON_CLASS =
  'flex h-5 items-center gap-1 rounded-md px-1 text-[0.625rem] font-bold text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-40'

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
  const copy = t.artifactPane

  if (total < 2) {
    return null
  }

  return (
    <div className="flex items-center gap-0.5 text-[0.625rem] font-bold text-muted-foreground">
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
      <span className="tabular-nums">{copy.versionOf(current + 1, total)}</span>
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
    </div>
  )
}

export function ArtifactPane({ artifactId }: { artifactId: string }) {
  const { t } = useI18n()
  const copy = t.artifactPane
  const registry = useStore($artifactRegistry)
  const versionSelection = useStore($artifactVersionSelection)
  // View mode is per-pane, ephemeral: renderable artifacts open in preview.
  const [userMode, setUserMode] = useState<ArtifactViewMode | null>(null)

  // Reset the explicit mode when the pane is reused for another artifact.
  useEffect(() => {
    setUserMode(null)
  }, [artifactId])

  const record = useMemo<ArtifactRecord | null>(() => {
    for (const records of Object.values(registry)) {
      const found = records.find(candidate => candidate.id === artifactId)

      if (found) {
        return found
      }
    }

    return null
  }, [artifactId, registry])

  if (!record) {
    return <PreviewEmptyState body={copy.missingBody} title={copy.missingTitle} />
  }

  const isRenderable = record.kind === 'html' || record.kind === 'svg'
  const versionIndex = Math.min(versionSelection[artifactId] ?? record.versions.length - 1, record.versions.length - 1)
  const version = record.versions[versionIndex]!
  const isCurrentVersion = versionIndex >= record.versions.length - 1
  const mode: ArtifactViewMode = isRenderable ? (userMode ?? 'preview') : 'source'
  const downloadName = artifactDownloadName(record.kind, record.language, record.title)

  const modeLabel: Record<ArtifactViewMode, string> = {
    preview: copy.modePreview,
    source: copy.modeSource
  }

  return (
    <div className="flex h-full flex-col overflow-hidden bg-transparent">
      <div className="flex h-7 shrink-0 items-center gap-3 border-b border-border/40 px-3">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <VersionStepper
            current={versionIndex}
            onSelect={index => selectArtifactVersion(artifactId, index)}
            total={record.versions.length}
          />
          {!isCurrentVersion && (
            <button
              className="text-[0.625rem] font-bold text-muted-foreground underline decoration-current/25 underline-offset-4 transition-colors hover:text-foreground"
              onClick={() => selectArtifactVersion(artifactId, record.versions.length - 1)}
              type="button"
            >
              {copy.latest}
            </button>
          )}
        </div>
        {isRenderable &&
          (['preview', 'source'] as const).map(candidate => (
            <button
              className={cn(
                'text-[0.625rem] font-bold underline-offset-4 transition-colors',
                candidate === mode
                  ? 'text-foreground underline decoration-current/30'
                  : 'text-muted-foreground hover:text-foreground'
              )}
              key={candidate}
              onClick={() => setUserMode(candidate)}
              type="button"
            >
              {modeLabel[candidate]}
            </button>
          ))}
        <div className="flex items-center gap-0.5">
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
              onClick={() => downloadTextFile(downloadName, version.content, MIME_BY_KIND[record.kind])}
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
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-hidden">
        {mode === 'preview' && isRenderable ? (
          <ArtifactLivePreview content={version.content} kind={record.kind} title={record.title} />
        ) : (
          <ArtifactSourceView language={record.kind === 'html' ? 'html' : record.language} text={version.content} />
        )}
      </div>
    </div>
  )
}
