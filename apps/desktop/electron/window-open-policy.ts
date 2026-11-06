/**
 * Window-open policy for every BrowserWindow's webContents.
 *
 * Every external URL the desktop opens on purpose goes through the audited
 * `hermes:openExternal` IPC channel (`openExternalUrl` in main.ts: http/https/
 * mailto allowlist, guarded file:). The `window.open` / `target=_blank` path
 * that reaches `setWindowOpenHandler` is therefore only ever driven by content
 * we did NOT initiate — most dangerously untrusted HTML in sandboxed
 * `allow-scripts` iframes (artifact previews, inline preview directives).
 *
 * GHSA-9f4c-93c8-jc8g (CVE-2026-70608): a sandboxed iframe without
 * `allow-popups` and without a user gesture can still reach this handler via
 * the OpenURL navigation path. If the handler opens `details.url` as a side
 * effect, a malicious artifact forces the user's OS browser to an attacker URL.
 * There is no fixed Electron 40.x, so the defence lives here regardless of the
 * pin: deny every request and never open a URL from this handler.
 *
 * The ONE, narrow exception (#91612): the embedded Skills Hub picker — whose
 * origin is pinned in hub-iframe-policy.ts — shows documentation and install
 * links that must open in the OS browser. For it, and ONLY it, the handler
 * delegates an http/https/mailto URL to the same audited `openExternalUrl`.
 * The delegation is a SIDE EFFECT of the deny decision: no window is ever
 * created from this path, and a frame whose origin is not EXACTLY the hub
 * origin (an opaque sandboxed frame reports `null`) triggers no side effect at
 * all, preserving the CVE-2026-70608 posture for every other guest.
 */

import { isHermesHubExternalUrl, isHermesHubOrigin } from './hub-iframe-policy'

export interface WindowOpenRequestLike {
  url: string
}

export interface WindowOpenDecision {
  action: 'deny'
}

/** How the handler learns the requesting frame's origin (dependency-injected
 *  so the policy unit-tests without Electron). */
export interface TrustedWindowOpenOptions {
  /** Origin of the frame that called window.open (Electron 40: the focused
   *  frame at the time of the open request). */
  getOpenerOrigin: (details: WindowOpenRequestLike) => string | null | undefined
  /** The audited external opener from main.ts; must never create a window. */
  openExternalUrl: (url: string) => unknown
}

/**
 * `origin` only — a denied URL can carry query credentials, signed-URL tokens
 * or attacker-controlled text, none of which belongs in a persisted log.
 */
export function describeDeniedUrl(url: string): string {
  try {
    const parsed = new URL(url)

    return parsed.origin === 'null' ? parsed.protocol : parsed.origin
  } catch {
    return '<unparseable>'
  }
}

/**
 * Build a `setWindowOpenHandler` callback that denies unconditionally.
 * `onDenied` is logging-only and receives the sanitized origin; a throwing
 * observer must not be able to change the decision or the side effect.
 * With `trusted` set, hub-origin openers additionally get their http/https/
 * mailto URL delegated to the audited external opener — still DENIED as a
 * window, never a popup.
 */
export function createWindowOpenHandler(
  onDenied?: (origin: string) => void,
  trusted?: TrustedWindowOpenOptions
): (details: WindowOpenRequestLike) => WindowOpenDecision {
  return details => {
    const openerOrigin = trusted ? safeOpenerOrigin(trusted, details) : null

    if (trusted && isHermesHubOrigin(openerOrigin) && isHermesHubExternalUrl(details.url)) {
      try {
        trusted.openExternalUrl(details.url)
      } catch {
        // A failed external open is logged by the opener itself; it must not
        // change the decision here either.
      }
    }

    try {
      onDenied?.(describeDeniedUrl(details.url))
    } catch {
      // observer failure is not a reason to reconsider the decision
    }

    return { action: 'deny' }
  }
}

/** The origin probe must never throw into the handler; unknown means deny. */
function safeOpenerOrigin(
  trusted: TrustedWindowOpenOptions,
  details: WindowOpenRequestLike
): string | null | undefined {
  try {
    return trusted.getOpenerOrigin(details)
  } catch {
    return null
  }
}
