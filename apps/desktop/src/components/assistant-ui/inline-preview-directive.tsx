import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { PreviewAttachment } from '@/components/chat/preview-attachment'
import { ScrollGate } from '@/components/assistant-ui/embeds/scroll-gate'
import { useIsDark } from '@/components/assistant-ui/embeds/use-is-dark'
import { useI18n } from '@/i18n'
import { isRemoteGateway } from '@/lib/media'
import { localPreviewTarget } from '@/lib/local-preview'

/**
 * `::preview{file="…"}` — a workspace HTML file rendered LIVE inside the
 * assistant message, bb-inline-vis style. A sandboxed iframe with an opaque
 * origin (`sandbox="allow-scripts"`, deliberately no `allow-same-origin`):
 * scripts run, but the document cannot reach the app, its cookies, storage,
 * or the bridge. The doc arrives via `srcdoc` from a bridge file read, so
 * single-file HTML (what agents generate) is fully live; relative sibling
 * assets don't resolve in an opaque origin — the header's rail affordance
 * covers multi-file apps with the full webview preview.
 *
 * Non-HTML targets and remote gateways (no local file access) fall back to
 * the standard preview-attachment card rather than a broken frame.
 */

const MIN_HEIGHT = 120
const MAX_HEIGHT = 1200
const DEFAULT_HEIGHT = 280

export function directiveFrameHeight(raw: string | undefined): number {
  if (!raw) {
    return DEFAULT_HEIGHT
  }

  const parsed = Number(raw)

  if (!Number.isInteger(parsed)) {
    return DEFAULT_HEIGHT
  }

  return Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, parsed))
}

const HTML_FILE_RE = /\.(?:html?|xhtml)$/i

export function InlinePreviewDirective({
  attrs,
  streaming
}: {
  attrs: Readonly<Record<string, string>>
  streaming: boolean
}) {
  const file = attrs.file ?? ''

  // Not renderable inline: hand the whole leaf to the classic card. Remote
  // gateways lack a local-file door; non-HTML has nothing to frame.
  if (!file || isRemoteGateway() || !HTML_FILE_RE.test(file)) {
    return file ? <PreviewAttachment source="explicit-link" target={file} /> : null
  }

  return <InlineHtmlFrame file={file} height={directiveFrameHeight(attrs.height)} streaming={streaming} />
}

function InlineHtmlFrame({ file, height, streaming }: { file: string; height: number; streaming: boolean }) {
  const { t } = useI18n()
  const cwd = useStore(useSessionView().$cwd)
  const isDark = useIsDark()
  const [doc, setDoc] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)

  // Resolve against THIS session's cwd (the file was written by its agent).
  const resolved = localPreviewTarget(file, cwd || undefined)
  const path = resolved?.path ?? null

  useEffect(() => {
    // Wait for turn settle: mid-stream the file is often mid-write, and a
    // half-written srcdoc renders as garbage that never self-corrects.
    if (!path || streaming) {
      return
    }

    let alive = true

    void Promise.resolve(window.hermesDesktop?.readFileText(path))
      .then(result => {
        if (!alive) {
          return
        }

        if (!result || result.binary || !result.text) {
          setFailed(true)
        } else {
          setDoc(result.text)
        }
      })
      .catch(() => alive && setFailed(true))

    return () => {
      alive = false
    }
  }, [path, streaming])

  if (!path || failed) {
    return <PreviewAttachment source="explicit-link" target={file} />
  }

  return (
    <span className="my-2 grid w-full max-w-160 gap-2">
      {doc === null ? (
        <span
          className="grid w-full animate-pulse place-items-center rounded-lg border border-(--ui-stroke-tertiary) text-[0.75rem] text-muted-foreground"
          style={{ height }}
        >
          {t.preview.opening}
        </span>
      ) : (
        <span
          className="relative block w-full overflow-hidden rounded-lg border border-(--ui-stroke-tertiary)"
          style={{ height }}
        >
          <iframe
            className="absolute inset-0 size-full border-0 bg-transparent"
            loading="lazy"
            sandbox="allow-scripts"
            srcDoc={doc}
            style={{ colorScheme: isDark ? 'dark' : 'light' }}
            title={file}
          />
          <ScrollGate />
        </span>
      )}
      <PreviewAttachment source="explicit-link" target={file} />
    </span>
  )
}
