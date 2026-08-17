import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

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
 * HEIGHT IS CONTENT-DRIVEN. The opaque origin means the parent can't measure
 * the document, but we own the srcdoc string — so a measuring script is
 * injected that posts the document's scrollHeight up via postMessage (tagged
 * with a per-mount token) on load and on every resize, and the frame tracks
 * it within the clamp band. Full-viewport pages (100vh) measure exactly the
 * height they're given, so they settle at the default instead of looping. An
 * explicit `height="480"` attribute opts out of auto-sizing entirely.
 *
 * Non-HTML targets and remote gateways (no local file access) fall back to
 * the standard preview-attachment card rather than a broken frame.
 */

const MIN_HEIGHT = 120
const MAX_HEIGHT = 1200
const DEFAULT_HEIGHT = 280
/** Ignore sub-pixel/rounding churn so a vh-sized page can't oscillate. */
const RESIZE_TOLERANCE = 4

export function directiveFrameHeight(raw: string | undefined): number | null {
  if (!raw) {
    return null
  }

  const parsed = Number(raw)

  if (!Number.isInteger(parsed)) {
    return null
  }

  return Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, parsed))
}

const SIZE_MESSAGE_TYPE = 'hermes-inline-preview-size'

/** The script injected into the srcdoc that reports content height to the
 *  parent. Runs inside the opaque origin, so postMessage is its only door —
 *  it can say "I am N pixels tall" and nothing else. */
export function measurementScript(token: string): string {
  return (
    '<script>(function(){var t=' +
    JSON.stringify(token) +
    ';var last=0;function post(){var d=document.documentElement;var b=document.body;' +
    'var h=Math.max(d?d.scrollHeight:0,b?b.scrollHeight:0);' +
    'if(Math.abs(h-last)>1){last=h;parent.postMessage({type:' +
    JSON.stringify(SIZE_MESSAGE_TYPE) +
    ',token:t,height:h},"*")}}' +
    'if(typeof ResizeObserver==="function"){var ro=new ResizeObserver(post);' +
    'ro.observe(document.documentElement);if(document.body)ro.observe(document.body)}' +
    'addEventListener("load",post);post()})()</script>'
  )
}

/** Inject the measuring script into a document string — before `</body>`
 *  when present so it runs after the page's own markup, appended otherwise. */
export function withMeasurement(doc: string, token: string): string {
  const script = measurementScript(token)
  const bodyClose = /<\/body\s*>/i.exec(doc)

  if (bodyClose) {
    return doc.slice(0, bodyClose.index) + script + doc.slice(bodyClose.index)
  }

  return doc + script
}

/** Parse a size report from the frame. Null unless it is OUR message type,
 *  carries OUR token, and holds a sane finite height — anything inside the
 *  sandbox can postMessage, so everything is validated before it moves the
 *  layout. Clamped to the band. */
export function frameHeightFromMessage(data: unknown, token: string): number | null {
  if (typeof data !== 'object' || data === null) {
    return null
  }

  const message = data as { type?: unknown; token?: unknown; height?: unknown }

  if (message.type !== SIZE_MESSAGE_TYPE || message.token !== token || typeof message.height !== 'number') {
    return null
  }

  if (!Number.isFinite(message.height) || message.height <= 0) {
    return null
  }

  return Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, Math.round(message.height)))
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

  return <InlineHtmlFrame file={file} fixedHeight={directiveFrameHeight(attrs.height)} streaming={streaming} />
}

function InlineHtmlFrame({
  file,
  fixedHeight,
  streaming
}: {
  file: string
  /** Explicit `height` attribute — opts out of content-driven sizing. */
  fixedHeight: number | null
  streaming: boolean
}) {
  const { t } = useI18n()
  const cwd = useStore(useSessionView().$cwd)
  const isDark = useIsDark()
  const [doc, setDoc] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  const [measured, setMeasured] = useState<number | null>(null)
  const heightRef = useRef<number>(fixedHeight ?? DEFAULT_HEIGHT)

  // One token per mount: the message listener only trusts reports from the
  // document THIS mount injected, so two previews in one transcript (or a
  // hostile page inventing messages) can't move each other's frames.
  const token = useMemo(() => Math.random().toString(36).slice(2), [])

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

  useEffect(() => {
    if (fixedHeight !== null) {
      return
    }

    const onMessage = (event: MessageEvent) => {
      const next = frameHeightFromMessage(event.data, token)

      if (next !== null && Math.abs(next - heightRef.current) > RESIZE_TOLERANCE) {
        heightRef.current = next
        setMeasured(next)
      }
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [fixedHeight, token])

  const framedDoc = useMemo(() => (doc === null ? null : withMeasurement(doc, token)), [doc, token])

  if (!path || failed) {
    return <PreviewAttachment source="explicit-link" target={file} />
  }

  const height = fixedHeight ?? measured ?? DEFAULT_HEIGHT

  return (
    <span className="my-2 grid w-full max-w-160 gap-2">
      {framedDoc === null ? (
        <span
          className="grid w-full animate-pulse place-items-center rounded-lg border border-(--ui-stroke-tertiary) text-[0.75rem] text-muted-foreground"
          style={{ height }}
        >
          {t.preview.opening}
        </span>
      ) : (
        <span
          className="relative block w-full overflow-hidden rounded-lg border border-(--ui-stroke-tertiary) transition-[height] duration-200"
          style={{ height }}
        >
          <iframe
            className="absolute inset-0 size-full border-0 bg-transparent"
            loading="lazy"
            sandbox="allow-scripts"
            srcDoc={framedDoc}
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
