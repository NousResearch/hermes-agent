import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useSessionView } from '@/app/chat/session-view'
import { useIsDark } from '@/components/assistant-ui/embeds/use-is-dark'
import { PreviewAttachment } from '@/components/chat/preview-attachment'
import { readDesktopFileText } from '@/lib/desktop-fs'
import { localPreviewTarget } from '@/lib/local-preview'

/**
 * `::preview{file="…"}` — a workspace HTML file rendered LIVE inside the
 * assistant message. A sandboxed iframe with an opaque origin
 * (`sandbox="allow-scripts"`, deliberately no `allow-same-origin`): scripts
 * run and the widget is fully interactive, but the document cannot reach the
 * app, its cookies, storage, or the bridge. The doc arrives via `srcdoc`
 * from a bridge file read, so single-file HTML (what agents generate) is
 * fully live; relative sibling assets don't resolve in an opaque origin.
 *
 * SIZE IS CONTENT-DRIVEN. The opaque origin means the parent can't measure
 * the document, but we own the srcdoc string — an injected script posts the
 * content's size up via postMessage (tagged with a per-mount token). Height
 * tracks live within the clamp band; the frame fills the message column and
 * tracks its width live, never shrinking below the measured intrinsic floor
 * (so %-width children can't spiral toward zero). A `height="480"`
 * attribute only sets the starting height — measurement always wins.
 *
 * NATIVE BY DEFAULT. A theme prelude injects first: the app's resolved
 * theme tokens under friendly names (--foreground, --muted-foreground,
 * --accent, --border, --card), the app's color scheme, the app font, zero
 * body margin/padding, and a transparent background — so widget-shaped
 * content reads as part of the app. The page's own styles override all of
 * it, so a full page keeps its own design.
 *
 * WIDGETS TALK BACK OFF-SCREEN. `window.hermes.send(prompt)` (or declarative
 * `data-hermes-send` on any clickable element) routes the prompt through the
 * composer's send path as a user turn typed `display_kind=hidden`: the agent
 * wakes and the durable row exists (context, resume, audit via the DB), but
 * no bubble renders — the widget updating is the visible response. Token-
 * gated, length-capped, throttled to human speed.
 *
 * Non-HTML targets and remote gateways (no local file access) fall back to
 * the standard preview-attachment card rather than a broken frame.
 */

const MIN_HEIGHT = 120
const MAX_HEIGHT = 1200
const DEFAULT_HEIGHT = 280
/** Ignore sub-pixel/rounding churn so a vh-sized page can't oscillate. */
const RESIZE_TOLERANCE = 4

/** Resolve the frame width from the content's intrinsic width and the live
 *  container width. Fills the transcript like text does: grows with the
 *  container, never shrinks below the measured intrinsic floor (so %-width
 *  children can't spiral toward zero), and preserves a wide intrinsic span
 *  for CSS `max-w-full` to cap to the column.
 *  Null when nothing is measured yet — the caller falls back to `100%`. */
export function resolvePreviewFrameWidth(
  contentWidth: number | null,
  containerWidth: number | null
): number | null {
  if (contentWidth === null) {
    return containerWidth
  }

  if (containerWidth === null) {
    return contentWidth
  }

  // ponytail: full-width frame even for narrow widgets (transparent canvas
  // so narrow content still sits left); per-content fluid-vs-fixed detection
  // if a narrow frame chrome ever matters.
  return Math.max(contentWidth, containerWidth)
}

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
const INTENT_MESSAGE_TYPE = 'hermes-inline-preview-intent'

/** Prompt length cap for a widget intent — a sentence, not a payload dump. */
const MAX_INTENT_LENGTH = 500
/** One intent per frame per second; clicks are human-speed. */
const INTENT_THROTTLE_MS = 1000

/** The script that gives the widget its ONE voice: `hermes.send(prompt)`.
 *  Posts the prompt up tagged with the mount token; the parent validates,
 *  throttles, and routes it through the composer as a normal user message —
 *  the widget speaks WITH the user's voice, visibly, never silently. Also
 *  wires `data-hermes-send` so declarative HTML works with zero script:
 *  `<button data-hermes-send="get-price eth">ETH</button>`. */
export function intentScript(token: string): string {
  return (
    '<script>(function(){var t=' +
    JSON.stringify(token) +
    ';function send(p){if(typeof p!=="string"||!p.trim())return false;' +
    'parent.postMessage({type:' +
    JSON.stringify(INTENT_MESSAGE_TYPE) +
    ',token:t,prompt:p.slice(0,' +
    String(MAX_INTENT_LENGTH) +
    ')},"*");return true}' +
    'window.hermes={send:send};' +
    'addEventListener("click",function(e){var el=e.target&&e.target.closest?' +
    'e.target.closest("[data-hermes-send]"):null;' +
    'if(el)send(el.getAttribute("data-hermes-send")||"")},true)})()</script>'
  )
}

/** Parse a widget intent. Null unless it is OUR type with OUR token and a
 *  non-empty string prompt — same trust boundary as size reports, because
 *  this one turns into a user message. Trimmed and length-capped. */
export function intentFromMessage(data: unknown, token: string): string | null {
  if (typeof data !== 'object' || data === null) {
    return null
  }

  const message = data as { type?: unknown; token?: unknown; prompt?: unknown }

  if (message.type !== INTENT_MESSAGE_TYPE || message.token !== token || typeof message.prompt !== 'string') {
    return null
  }

  const prompt = message.prompt.trim().slice(0, MAX_INTENT_LENGTH)

  return prompt || null
}

/** Semantic tokens handed into the frame, resolved to concrete values from
 *  the LIVE theme. Friendly names, not internal ones — this is the contract
 *  reference HTML / skills write against (`var(--foreground)` etc.). */
const THEME_BRIDGE_TOKENS: Record<string, string> = {
  '--foreground': '--ui-text-primary',
  '--muted-foreground': '--ui-text-tertiary',
  '--accent': '--ui-accent',
  '--border': '--ui-stroke-tertiary',
  '--card': '--ui-bg-editor'
}

/** Resolve the bridge tokens + app font against the current document. */
export function collectThemeBridge(): { vars: Record<string, string>; font: string } {
  const vars: Record<string, string> = {}

  if (typeof document !== 'undefined') {
    const root = getComputedStyle(document.documentElement)

    for (const [alias, source] of Object.entries(THEME_BRIDGE_TOKENS)) {
      const value = root.getPropertyValue(source).trim()

      if (value) {
        vars[alias] = value
      }
    }
  }

  const font = typeof document === 'undefined' ? '' : getComputedStyle(document.body).fontFamily

  return { vars, font }
}

/**
 * The style prelude that makes an inline widget read as NATIVE: the app's
 * resolved theme tokens as CSS vars, the app's color scheme (so UA controls,
 * scrollbars, form defaults, and `prefers-color-scheme` inside the frame
 * follow the app instead of the UA light default — a transparent background
 * alone does not do this, #95814), the app font, no margin, and a
 * transparent background so the widget sits directly on the chat surface.
 * Injected FIRST, so the page's own styles override every default here — a
 * full page that wants its own look keeps it, including its own
 * `color-scheme` declaration.
 */
export function themePrelude(vars: Record<string, string>, font: string, colorScheme: 'light' | 'dark'): string {
  const tokens = Object.entries(vars)
    .map(([name, value]) => `${name}:${value}`)
    .join(';')

  const fontRule = font ? `font-family:${font};` : ''

  return (
    `<style>:root{color-scheme:${colorScheme};${tokens}}` +
    `html,body{margin:0;padding:0;background:transparent;color:var(--foreground,inherit);${fontRule}}</style>`
  )
}

/** The script injected into the srcdoc that reports content size to the
 *  parent. Runs inside the opaque origin, so postMessage is its only door —
 *  it can say "I am N pixels" and nothing else. Height is the document
 *  scrollHeight; width is the union of the body children's boxes (intrinsic
 *  content width — the document itself always fills the viewport, so
 *  scrollWidth would just echo the frame back). */
export function measurementScript(token: string): string {
  return (
    '<script>(function(){var t=' +
    JSON.stringify(token) +
    ';var lastH=0,lastW=0;function post(){var d=document.documentElement;var b=document.body;' +
    'var h=Math.max(d?d.scrollHeight:0,b?b.scrollHeight:0);' +
    'var w=0;if(b){var kids=b.children;var L=Infinity,R=0;for(var i=0;i<kids.length;i++){' +
    'var r=kids[i].getBoundingClientRect();if(r.width===0&&r.height===0)continue;' +
    'if(r.left<L)L=r.left;if(r.right>R)R=r.right}' +
    'if(R>L)w=R-L}' +
    'w=Math.ceil(w);' +
    'if(Math.abs(h-lastH)>1||Math.abs(w-lastW)>1){lastH=h;lastW=w;parent.postMessage({type:' +
    JSON.stringify(SIZE_MESSAGE_TYPE) +
    ',token:t,height:h,width:w},"*")}}' +
    'if(typeof ResizeObserver==="function"){var ro=new ResizeObserver(post);' +
    'ro.observe(document.documentElement);if(document.body)ro.observe(document.body)}' +
    'addEventListener("load",post);post()})()</script>'
  )
}

/** Assemble the srcdoc: theme prelude first (so the page's own styles win),
 *  then the measuring + intent scripts before `</body>` when present so they
 *  run after the page's own markup, appended otherwise. */
export function withInlineChrome(doc: string, token: string, prelude: string): string {
  const script = measurementScript(token) + intentScript(token)
  const bodyClose = /<\/body\s*>/i.exec(doc)
  const framed = bodyClose ? doc.slice(0, bodyClose.index) + script + doc.slice(bodyClose.index) : doc + script

  return prelude + framed
}

export interface FrameSizeReport {
  height: number
  /** Intrinsic content width, 0 when unmeasurable. */
  width: number
}

/** Parse a size report from the frame. Null unless it is OUR message type,
 *  carries OUR token, and holds a sane finite height — anything inside the
 *  sandbox can postMessage, so everything is validated before it moves the
 *  layout. Height clamped to the band; width sanitized but uncapped (the
 *  frame caps it against the column at render). */
export function frameSizeFromMessage(data: unknown, token: string): FrameSizeReport | null {
  if (typeof data !== 'object' || data === null) {
    return null
  }

  const message = data as { type?: unknown; token?: unknown; height?: unknown; width?: unknown }

  if (message.type !== SIZE_MESSAGE_TYPE || message.token !== token || typeof message.height !== 'number') {
    return null
  }

  if (!Number.isFinite(message.height) || message.height <= 0) {
    return null
  }

  const width =
    typeof message.width === 'number' && Number.isFinite(message.width) && message.width > 0
      ? Math.round(message.width)
      : 0

  return {
    height: Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, Math.round(message.height))),
    width
  }
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

  // Not renderable inline: hand the leaf to the classic card. Non-HTML has
  // nothing to frame. (Remote gateways used to bail here too — that predates
  // the mode-aware fs bridge; the frame now reads through readDesktopFileText,
  // which fetches over the authenticated /api/fs bridge in remote mode, so a
  // URL connection — including a same-machine `hermes serve` — renders live.)
  if (!file || !HTML_FILE_RE.test(file)) {
    return file ? <PreviewAttachment target={file} /> : null
  }

  return <InlineHtmlFrame file={file} initialHeight={directiveFrameHeight(attrs.height)} streaming={streaming} />
}

function InlineHtmlFrame({
  file,
  initialHeight,
  streaming
}: {
  file: string
  /** `height` attribute — the starting height only; measurement overrides. */
  initialHeight: number | null
  streaming: boolean
}) {
  const cwd = useStore(useSessionView().$cwd)
  const isDark = useIsDark()
  const colorScheme = isDark ? 'dark' : 'light'
  const [doc, setDoc] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  const [measured, setMeasured] = useState<number | null>(null)
  const [contentWidth, setContentWidth] = useState<number | null>(null)
  const [containerWidth, setContainerWidth] = useState<number | null>(null)
  const containerRef = useRef<HTMLSpanElement | null>(null)

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

    void Promise.resolve(readDesktopFileText(path))
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
    // Human-speed gate on widget intents. A closure local, not state: it's
    // a rate limiter read inside the handler, never rendered.
    let lastIntentAt = 0

    const onMessage = (event: MessageEvent) => {
      const intent = intentFromMessage(event.data, token)

      if (intent !== null) {
        const now = Date.now()

        if (now - lastIntentAt >= INTENT_THROTTLE_MS) {
          lastIntentAt = now
          // Off-screen: the prompt reaches the agent as a normal user turn
          // through the composer's own send path (steer/queue rules apply),
          // but the row is typed hidden — no bubble, no UI space. The widget
          // updating IS the visible response.
          requestComposerSubmit(intent, { target: 'active', displayKind: 'hidden' })
        }

        return
      }

      const next = frameSizeFromMessage(event.data, token)

      if (next === null) {
        return
      }

      // Functional updates so the comparisons read current state without a
      // shadow ref: same-value sets bail out in React, and the tolerance
      // keeps a vh-sized page (which measures what it's given) from
      // oscillating.
      setMeasured(prev =>
        Math.abs(next.height - (prev ?? initialHeight ?? DEFAULT_HEIGHT)) > RESIZE_TOLERANCE ? next.height : prev
      )

      // Intrinsic floor grows monotonically: later reports can raise it when
      // content truly widens, but never lower it — %-width children reflow
      // narrower every time the frame shrinks, so tracking shrinks live
      // would spiral toward zero. Live growth comes from the container
      // measurement below instead.
      if (next.width > 0) {
        setContentWidth(prev => (prev === null ? next.width : Math.max(prev, next.width)))
      }
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [initialHeight, token])

  useEffect(() => {
    // Live container width so the frame fills the transcript and tracks
    // window resizes; the intrinsic floor above keeps %-width pages from
    // oscillating.
    const el = containerRef.current

    if (!el || typeof ResizeObserver === 'undefined') {
      return
    }

    setContainerWidth(el.clientWidth || null)

    const ro = new ResizeObserver(entries => {
      const next = entries[0]?.contentRect.width

      if (typeof next === 'number' && Number.isFinite(next) && next > 0) {
        setContainerWidth(prev =>
          prev === null || Math.abs(next - prev) > RESIZE_TOLERANCE ? Math.round(next) : prev
        )
      }
    })

    ro.observe(el)

    return () => ro.disconnect()
  }, [])

  // Rebuild the srcdoc when the color scheme changes so its native controls and
  // transparent canvas stay aligned with the app.
  const framedDoc = useMemo(() => {
    if (doc === null) {
      return null
    }

    const { vars, font } = collectThemeBridge()

    return withInlineChrome(doc, token, themePrelude(vars, font, colorScheme))
  }, [colorScheme, doc, token])

  if (!path || failed) {
    return <PreviewAttachment target={file} />
  }

  const height = measured ?? initialHeight ?? DEFAULT_HEIGHT
  // Fill the message column like text does; a wide intrinsic span is kept
  // so fluid pages stay full-bleed, with `max-w-full` capping overflow.
  const width = resolvePreviewFrameWidth(contentWidth, containerWidth)

  return (
    <span ref={containerRef} className="my-2 block w-full max-w-full">
      {framedDoc === null ? (
        <span
          className="block w-full animate-pulse rounded-md bg-[color-mix(in_srgb,currentColor_4%,transparent)]"
          style={{ height }}
        />
      ) : (
        <span
          className="relative block max-w-full transition-[height] duration-200"
          style={{ height, width: width ?? '100%' }}
        >
          <iframe
            className="absolute inset-0 size-full border-0 bg-transparent"
            loading="lazy"
            sandbox="allow-scripts"
            srcDoc={framedDoc}
            style={{ colorScheme }}
            title={file}
          />
        </span>
      )}
    </span>
  )
}
