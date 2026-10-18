import type * as React from 'react'

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

function hasAllowedScheme(src: string): boolean {
  try {
    return ALLOWED_SRC_SCHEMES.has(new URL(src).protocol)
  } catch {
    return false
  }
}
