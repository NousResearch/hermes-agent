import type * as React from 'react'
import { useCallback, useEffect, useLayoutEffect, useRef } from 'react'

import { htmlFromDataUrl } from '@/lib/browser-guest-document'
import { browserBootstrap, endpointUrl } from '@/lib/browser-transport'
import { isBrowserHostedDesktop } from '@/lib/platform'
import { cn } from '@/lib/utils'

/** The sandbox posture guest content gets by default: scripts run, the frame
 *  has an OPAQUE origin (no reach into the app, its storage, or the bridge). */
export const SANDBOXED_FRAME_DEFAULT_SANDBOX = 'allow-scripts'

/** The only sandbox tokens a caller may add. An ALLOWLIST, not a blocklist:
 *  HTML defines the attribute as a case-insensitive token set and browsers
 *  lower-case each token before matching it, so a blocklist is only as good as
 *  its spelling — `ALLOW-SAME-ORIGIN` sailed past a case-sensitive set and the
 *  browser honoured it. With an allowlist an unrecognised token is dropped
 *  instead of forwarded, so the posture holds by construction.
 *
 *  What is deliberately NOT here: `allow-same-origin` (re-opens cookies and
 *  storage for the embedded origin — the opaque origin IS the containment) and
 *  the navigation / popup / modals family (a guest driving the host window).
 *  A caller wanting one of those needs a real review, not a prop.
 *
 *  `allow-downloads` IS here: it writes to the user's disk from guest content,
 *  which is why it is not in the default posture — but it is a user-visible
 *  download rather than an escape from the sandbox, and a plugin embedding a
 *  document viewer legitimately needs it. */
const ALLOWED_SANDBOX_TOKENS = new Set([
  'allow-downloads',
  'allow-forms',
  'allow-orientation-lock',
  'allow-pointer-lock',
  'allow-presentation',
  'allow-scripts'
])

/** Keep only the allowed tokens from a caller's sandbox string (pure, for
 *  tests). Case-insensitive, because the attribute is: every token is
 *  lower-cased before the membership test, so no spelling of a forbidden token
 *  survives. Empty result falls back to the default posture rather than an
 *  unsandboxed frame — a frame with NO sandbox attribute is fully privileged. */
export function sanitizeFrameSandbox(sandbox: string | undefined): string {
  const tokens = (sandbox ?? '')
    .split(/\s+/)
    .map(token => token.trim().toLowerCase())
    .filter(token => token && ALLOWED_SANDBOX_TOKENS.has(token))

  return tokens.length > 0 ? [...new Set(tokens)].join(' ') : SANDBOXED_FRAME_DEFAULT_SANDBOX
}

/** Schemes an embed may point at. `file:` would read the user's disk into a
 *  scripted frame, `blob:`/`javascript:` are host-realm content, and a
 *  relative path resolves against the app's own origin. */
const ALLOWED_SRC_SCHEMES = new Set(['data:', 'http:', 'https:'])

/** Explicit allowlist, NOT `ComponentProps<'iframe'>`: an iframe's attribute
 *  surface is how a frame gains authority — `allow` delegates
 *  Permissions-Policy (the electron permission handlers grant media capture
 *  without checking which frame asked), `srcdoc` replaces `src` with caller
 *  markup, `name` makes the frame a navigation target, `csp`/`credentialless`
 *  change the guest realm. None of those are props here, and no spread reaches
 *  the element, so a plugin cannot smuggle one through a cast either. */
export interface SandboxedFrameProps {
  className?: string
  onError?: React.ReactEventHandler<HTMLIFrameElement>
  onLoad?: React.ReactEventHandler<HTMLIFrameElement>
  ref?: React.Ref<HTMLIFrameElement>
  /** Extra sandbox tokens; filtered through the allowlist above. */
  sandbox?: string
  /** Absolute `http(s):` or `data:` URL to embed. Anything else renders nothing. */
  src: string
  style?: React.CSSProperties
  /** Accessible title (required — an untitled frame is unlabelled in the a11y tree). */
  title: string
}

/**
 * The sanctioned way for a plugin to embed external web content. Renders a
 * sandboxed iframe with the app's guest-content posture — never a raw
 * Electron `<webview>` (which would land on the app's own `persist:` preview
 * partition, sharing its cookies and storage).
 *
 * `sandbox` may add tokens from the allowlist (`allow-forms`,
 *  `allow-downloads`, …); everything else — `allow-same-origin`, the
 *  navigation/popup/modals family, and any token this primitive does not know —
 *  is dropped, case-insensitively. See {@link sanitizeFrameSandbox}.
 *
 * `loading` and `referrerPolicy` are the primitive's posture, not props
 * (`unsafe-url` would leak the app's origin to the embedded site, which is the
 * point of `no-referrer`).
 */
export function SandboxedFrame({ className, onError, onLoad, ref, sandbox, src, style, title }: SandboxedFrameProps) {
  if (!hasAllowedScheme(src)) {
    console.warn(`[SandboxedFrame] refusing to embed ${JSON.stringify(src)}: only http(s): and data: URLs are allowed`)

    return null
  }

  const html = isBrowserHostedDesktop() ? htmlFromDataUrl(src) : null

  if (html !== null) {
    return (
      <BrowserGuestFrame
        className={className}
        html={html}
        key={sanitizeFrameSandbox(sandbox)}
        onError={onError}
        onLoad={onLoad}
        ref={ref}
        sandbox={sandbox}
        style={style}
        title={title}
      />
    )
  }

  return (
    <iframe
      className={cn('size-full border-0 bg-transparent', className)}
      loading="lazy"
      onError={onError}
      onLoad={onLoad}
      ref={ref}
      referrerPolicy="no-referrer"
      sandbox={sanitizeFrameSandbox(sandbox)}
      src={src}
      style={style}
      title={title}
    />
  )
}

interface BrowserGuestFrameProps extends Omit<SandboxedFrameProps, 'src'> {
  html: string
}

// A shell's own load is taken at the document in the capture phase (load
// events never reach the window), before React or a consumer's handler sees
// it (inside a shadow root, the frame's own handler takes it instead); guest
// loads are dispatched on the shell's frame as marked load events.
// Both keep working while an <Activity> or Suspense boundary hides the frame
// and detaches its ref and effects. A frame mounted hidden has no handler yet:
// its shell load is only recorded, and the frame starts when first shown.
const shellLoads = new WeakMap<HTMLIFrameElement, (frame: HTMLIFrameElement) => void>()
const loadedShells = new WeakSet<HTMLIFrameElement>()
const shellFrames = new WeakMap<object, HTMLIFrameElement>()
const guestLoadEvents = new WeakSet<Event>()
let listening = false

function listenForGuestShells() {
  if (listening) {
    return
  }

  listening = true
  document.addEventListener(
    'load',
    event => {
      const frame = event.target

      if (frame instanceof HTMLIFrameElement && frame.hasAttribute('data-guest-shell') && !guestLoadEvents.has(event)) {
        event.stopImmediatePropagation()
        loadedShells.add(frame)
        shellLoads.get(frame)?.(frame)
      }
    },
    true
  )
  window.addEventListener('message', event => {
    const frame = event.source && event.data?.type === 'hermes-guest-loaded' ? shellFrames.get(event.source) : undefined

    if (frame?.isConnected && frame.contentWindow === event.source) {
      const load = new Event('load')

      guestLoadEvents.add(load)
      frame.dispatchEvent(load)
    }
  })
}

/** Webapp only: a `data:text/html` document would inherit the page CSP, which
 *  never runs guest inline scripts. The gateway's guest shell is served with
 *  its own policy and the SDK allowlist as its sandbox ceiling; this frame's
 *  sandbox narrows it to the caller's tokens (plus `allow-scripts`, which the
 *  shell itself needs), and the guest document nested in the shell gets the
 *  caller's tokens exactly. The shell relays messages both ways, so
 *  `ref.current.contentWindow.postMessage` and `event.source ===
 *  ref.current.contentWindow` keep working, and `onLoad` fires once the guest
 *  document (not the shell) has loaded. */
function BrowserGuestFrame({ className, html, onError, onLoad, ref, sandbox, style, title }: BrowserGuestFrameProps) {
  const frame = useRef<HTMLIFrameElement | null>(null)
  const guestSandbox = sanitizeFrameSandbox(sandbox)
  // Before the first commit, so no shell can load unseen.
  listenForGuestShells()
  const lastDocument = useRef<{ html: string; sandbox: string } | null>(null)
  const src = endpointUrl('/assets/webapp-guest.html', browserBootstrap()?.basePath || '').href

  const send = useCallback(
    (node: HTMLIFrameElement) => {
      const target = node.contentWindow

      if (!target || (lastDocument.current?.html === html && lastDocument.current.sandbox === guestSandbox)) {
        return
      }

      target.postMessage({ type: 'hermes-preview-document', html, sandbox: guestSandbox }, '*')
      lastDocument.current = { html, sandbox: guestSandbox }
    },
    [html, guestSandbox]
  )

  const shellLoaded = useCallback(
    (node: HTMLIFrameElement) => {
      shellFrames.set(node.contentWindow!, node)
      lastDocument.current = null
      send(node)
    },
    [send]
  )

  // Registered again with each document, so a shell that loads while hidden
  // still gets the latest one this frame showed; a shell that loaded before
  // this frame was first shown is started now.
  useLayoutEffect(() => {
    const node = frame.current

    if (node) {
      shellLoads.set(node, shellLoaded)

      if (loadedShells.has(node) && shellFrames.get(node.contentWindow!) !== node) {
        shellLoaded(node)
      }
    }
  }, [shellLoaded])

  useEffect(() => {
    const node = frame.current

    if (node?.contentWindow && shellFrames.get(node.contentWindow) === node) {
      send(node)
    }
  }, [send])

  const attach = useCallback(
    (node: HTMLIFrameElement | null) => {
      frame.current = node
      const cleanup = typeof ref === 'function' ? ref(node) : undefined

      if (ref && typeof ref !== 'function') {
        ref.current = node
      }

      return () => {
        frame.current = null

        if (typeof cleanup === 'function') {
          cleanup()
        } else if (typeof ref === 'function') {
          ref(null)
        } else if (ref) {
          ref.current = null
        }
      }
    },
    [ref]
  )

  return (
    <iframe
      className={cn('size-full border-0 bg-transparent', className)}
      data-guest-shell=""
      loading="lazy"
      onError={onError}
      onLoad={event => {
        if (guestLoadEvents.has(event.nativeEvent)) {
          onLoad?.(event)

          return
        }

        // Only a shell load the document never saw (a load is not composed, so
        // inside a shadow root it stops there) reaches React.
        event.stopPropagation()
        loadedShells.add(event.currentTarget)
        shellLoaded(event.currentTarget)
      }}
      ref={attach}
      referrerPolicy="no-referrer"
      sandbox={sanitizeFrameSandbox(`${guestSandbox} allow-scripts`)}
      src={src}
      style={style}
      title={title}
    />
  )
}

function hasAllowedScheme(src: string): boolean {
  try {
    return ALLOWED_SRC_SCHEMES.has(new URL(src).protocol)
  } catch {
    return false
  }
}
