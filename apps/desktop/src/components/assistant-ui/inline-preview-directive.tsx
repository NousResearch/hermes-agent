import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useSessionView } from '@/app/chat/session-view'
import { PreviewAttachment } from '@/components/chat/preview-attachment'
import { useThemeEpoch } from '@/hooks/use-theme-epoch'
import { isReadFileErrorResult, readDesktopFileText } from '@/lib/desktop-fs'
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
 * tracks live within the clamp band; width adopts ONCE from the first
 * report, so a fixed-size widget shrink-wraps and sits left in the message
 * flow like an image, while a fluid page measures the full viewport and
 * stays column-wide. A `height="480"` attribute only sets the starting
 * height — measurement always wins.
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
 * gated, length-capped, throttled to human speed. Nothing is lost silently:
 * `send()` resolves a delivery ack, and an over-length, throttled, or
 * undeliverable intent resolves `{ ok: false, error }` instead of being
 * truncated or dropped behind the widget's back (#118973).
 *
 * Non-HTML targets and remote gateways (no local file access) fall back to
 * the standard preview-attachment card rather than a broken frame.
 */

const MIN_HEIGHT = 120
const MAX_HEIGHT = 1200
const DEFAULT_HEIGHT = 280
/** The transcript column cap the frame renders inside (`max-w-160` = 40rem). */
const MAX_COLUMN_WIDTH = 640
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
const INTENT_MESSAGE_TYPE = 'hermes-inline-preview-intent'
const INTENT_ACK_MESSAGE_TYPE = 'hermes-inline-preview-intent-ack'

/** Prompt length cap for a widget intent — a sentence, not a payload dump.
 *  Over-length intents are REJECTED back to the widget, never truncated: a
 *  sliced JSON payload reaches the agent as garbage while the widget thinks
 *  it sent. */
export const MAX_INTENT_LENGTH = 500
/** One intent per frame per second; clicks are human-speed. A faster intent
 *  is rejected with `retryAfterMs`, not dropped silently. */
export const INTENT_THROTTLE_MS = 1000
/** How long `hermes.send()` waits for the parent's ack before resolving
 *  `timeout` (the parent acks synchronously; this only fires if it's gone). */
const INTENT_ACK_TIMEOUT_MS = 5000

export type IntentError = 'invalid' | 'too_long' | 'throttled' | 'undelivered'

/** What `hermes.send()` resolves to inside the frame. `ok` means the prompt
 *  was handed to the owning composer's send path — not that the agent has
 *  answered. */
export type IntentAck = { ok: true } | { error: IntentError; maxLength?: number; ok: false; retryAfterMs?: number }

/** The script that gives the widget its ONE voice: `hermes.send(prompt)`.
 *  Posts the prompt up tagged with the mount token and a per-call id; the
 *  parent validates, throttles, routes it through the composer, and posts an
 *  ack back, so `send()` returns a Promise of an `IntentAck` — a widget can
 *  show "saved" only when it was. Also wires `data-hermes-send` so
 *  declarative HTML works with zero script:
 *  `<button data-hermes-send="get-price eth">ETH</button>`. */
export function intentScript(token: string): string {
  return (
    '<script>(function(){var t=' +
    JSON.stringify(token) +
    ';var n=0,w={};' +
    'addEventListener("message",function(e){var d=e.data;' +
    'if(e.source!==parent||!d||d.type!==' +
    JSON.stringify(INTENT_ACK_MESSAGE_TYPE) +
    '||d.token!==t||!w[d.id])return;' +
    'var r=w[d.id];delete w[d.id];r(d.ack)});' +
    'function send(p){if(typeof p!=="string"||!p.trim())return Promise.resolve({ok:false,error:"invalid"});' +
    'var id=++n;return new Promise(function(res){' +
    'var done=function(a){clearTimeout(k);res(a)};' +
    'var k=setTimeout(function(){if(w[id]){delete w[id];res({ok:false,error:"undelivered"})}},' +
    String(INTENT_ACK_TIMEOUT_MS) +
    ');w[id]=done;' +
    'parent.postMessage({type:' +
    JSON.stringify(INTENT_MESSAGE_TYPE) +
    ',token:t,id:id,prompt:p},"*")})}' +
    'window.hermes={send:send,maxLength:' +
    String(MAX_INTENT_LENGTH) +
    '};' +
    'addEventListener("click",function(e){var el=e.target&&e.target.closest?' +
    'e.target.closest("[data-hermes-send]"):null;' +
    'if(el)send(el.getAttribute("data-hermes-send")||"")},true)})()</script>'
  )
}

export type IntentDecision =
  | { id: number | null; kind: 'accept'; prompt: string }
  | { ack: Extract<IntentAck, { ok: false }>; id: number | null; kind: 'reject' }

/** The pure intent gate. Null unless it is OUR type with OUR token — same
 *  trust boundary as size reports, because an accepted intent turns into a
 *  user message; anything else is ignored without a reply. For our own
 *  messages every outcome is explicit: accept (trimmed, never truncated) or
 *  reject with a reason the frame gets back. `lastAcceptedAt` is the time
 *  of this frame's previous accepted intent (0 for none). */
export function evaluateIntent(
  data: unknown,
  token: string,
  now: number,
  lastAcceptedAt: number
): IntentDecision | null {
  if (typeof data !== 'object' || data === null) {
    return null
  }

  const message = data as { id?: unknown; prompt?: unknown; token?: unknown; type?: unknown }

  if (message.type !== INTENT_MESSAGE_TYPE || message.token !== token) {
    return null
  }

  const id = typeof message.id === 'number' && Number.isSafeInteger(message.id) ? message.id : null
  const prompt = typeof message.prompt === 'string' ? message.prompt.trim() : ''

  if (!prompt) {
    return { ack: { error: 'invalid', ok: false }, id, kind: 'reject' }
  }

  if (prompt.length > MAX_INTENT_LENGTH) {
    return { ack: { error: 'too_long', maxLength: MAX_INTENT_LENGTH, ok: false }, id, kind: 'reject' }
  }

  const elapsed = now - lastAcceptedAt

  if (lastAcceptedAt > 0 && elapsed < INTENT_THROTTLE_MS) {
    return { ack: { error: 'throttled', ok: false, retryAfterMs: INTENT_THROTTLE_MS - elapsed }, id, kind: 'reject' }
  }

  return { id, kind: 'accept', prompt }
}

/** The ack posted back into the frame for intent `id`. */
export function intentAckMessage(token: string, id: number, ack: IntentAck) {
  return { ack, id, token, type: INTENT_ACK_MESSAGE_TYPE }
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

/** The app's resolved light/dark appearance, read from the SAME attribute
 *  `applyTheme` (themes/context.tsx) sets the token values from — not the
 *  `.dark` class via `useIsDark()`, which is React state and can still hold
 *  the previous render's value for one paint after a real theme change. A
 *  direct read here means the tokens collected below and the color-scheme
 *  the frame is built with can never disagree (#123048). Mirrors the same
 *  fallback as `lib/selection-copy-colors.ts`'s `renderedMode()`. */
function resolvedColorScheme(): 'light' | 'dark' {
  if (typeof document === 'undefined') {
    return 'light'
  }

  const mode = document.documentElement.dataset.hermesMode

  if (mode === 'light' || mode === 'dark') {
    return mode
  }

  try {
    return document.defaultView?.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

/** Resolve the bridge tokens + app font + color scheme against the current
 *  document, all from one synchronous read so they can't drift apart. */
export function collectThemeBridge(): { vars: Record<string, string>; font: string; colorScheme: 'light' | 'dark' } {
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

  return { vars, font, colorScheme: resolvedColorScheme() }
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
  const themeEpoch = useThemeEpoch()
  // vars/font/colorScheme come from one collectThemeBridge() call so they can
  // never disagree with each other, even while this lags a repaint behind
  // `themeEpoch` (same trade-off use-is-dark.ts makes for the same reason).
  const [bridge, setBridge] = useState(collectThemeBridge)

  useEffect(() => setBridge(collectThemeBridge()), [themeEpoch])

  const [doc, setDoc] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  const [measured, setMeasured] = useState<number | null>(null)
  const [contentWidth, setContentWidth] = useState<number | null>(null)

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

        if (!result || isReadFileErrorResult(result)) {
          setFailed(true)

          return
        }

        if (result.binary || !result.text) {
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
      const decision = evaluateIntent(event.data, token, Date.now(), lastIntentAt)

      if (decision !== null) {
        let ack: IntentAck

        if (decision.kind === 'reject') {
          ack = decision.ack
        } else {
          lastIntentAt = Date.now()
          // Off-screen: the prompt reaches the agent as a normal user turn
          // through the composer's own send path (steer/queue rules apply),
          // but the row is typed hidden — no bubble, no UI space. The widget
          // updating IS the visible response. `false` means no visible
          // composer surface took it; the widget is told instead of assuming.
          ack = requestComposerSubmit(decision.prompt, { target: 'active', displayKind: 'hidden' })
            ? { ok: true }
            : { error: 'undelivered', ok: false }
        }

        // Reply only to the sender that proved it holds this mount's token.
        // The ack carries status, never app data; '*' because the sandboxed
        // frame's origin is opaque.
        const source = event.source as Window | null

        if (decision.id !== null && source) {
          source.postMessage(intentAckMessage(token, decision.id, ack), '*')
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

      // Width adopts ONCE, from the first report — measured at full column
      // width, so it is the content's intrinsic span. Tracking width live
      // would feedback-loop: %-width children reflow narrower every time
      // the frame shrinks, spiraling toward zero.
      if (next.width > 0) {
        setContentWidth(prev => prev ?? next.width)
      }
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [initialHeight, token])

  const { vars, font, colorScheme } = bridge

  // Rebuild the srcdoc when the bridge changes (a repaint or a skin swap,
  // whichever changed a token or the scheme) so its native controls and
  // transparent canvas stay aligned with the app.
  const framedDoc = useMemo(() => {
    if (doc === null) {
      return null
    }

    return withInlineChrome(doc, token, themePrelude(vars, font, colorScheme))
  }, [vars, font, colorScheme, doc, token])

  if (!path || failed) {
    return <PreviewAttachment target={file} />
  }

  const height = measured ?? initialHeight ?? DEFAULT_HEIGHT
  // Left-aligned in the message flow, like an image: the frame is only as
  // wide as its content (capped at the column). Fluid pages measure the
  // full viewport and stay full-bleed.
  const width = contentWidth !== null ? Math.min(contentWidth, MAX_COLUMN_WIDTH) : undefined

  return (
    <span className="my-2 block w-full max-w-160">
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
