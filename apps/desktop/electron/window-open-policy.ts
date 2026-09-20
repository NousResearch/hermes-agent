import { isHermesHubExternalUrl, isHermesHubOrigin } from './hub-iframe-policy'

/**
 * Window-open policy for every BrowserWindow's webContents.
 *
 * External URLs normally go through the audited `hermes:openExternal` IPC
 * channel. The one exception is the trusted Hermes Hub iframe: its `_blank`
 * links reach this handler, where the opener frame origin is checked before the
 * URL is handed to that same audited opener. All other popup requests stay
 * side-effect free, especially untrusted artifact-preview iframes.
 *
 * GHSA-9f4c-93c8-jc8g (CVE-2026-70608): a sandboxed iframe without
 * `allow-popups` and without a user gesture can still reach this handler via
 * the OpenURL navigation path. If the handler opens `details.url` as a side
 * effect, a malicious artifact forces the user's OS browser to an attacker URL.
 * There is no fixed Electron 40.x, so the defence lives here regardless of the
 * pin: Electron always denies popup creation; only the exact Hub origin may
 * delegate http/https/mailto links to the audited OS-browser opener.
 */

export interface WindowOpenRequestLike {
  url: string
}

export interface WindowOpenDecision {
  action: 'deny'
}

export interface TrustedWindowOpenOptions {
  getOpenerOrigin: () => string | undefined
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
 * observer must not be able to change the decision.
 */
export function createWindowOpenHandler(
  onDenied?: (origin: string) => void,
  trustedHub?: TrustedWindowOpenOptions
): (details: WindowOpenRequestLike) => WindowOpenDecision {
  return details => {
    if (trustedHub) {
      try {
        if (isHermesHubOrigin(trustedHub.getOpenerOrigin()) && isHermesHubExternalUrl(details.url)) {
          try {
            trustedHub.openExternalUrl(details.url)
          } catch {
            // The Electron popup remains denied even if external opening fails.
          }

          return { action: 'deny' }
        }
      } catch {
        // Opener inspection failed; fall through to the unconditional deny path.
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
