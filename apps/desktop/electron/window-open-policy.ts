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
 */

export interface WindowOpenRequestLike {
  url: string
}

export interface WindowOpenDecision {
  action: 'deny'
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
  onDenied?: (origin: string) => void
): (details: WindowOpenRequestLike) => WindowOpenDecision {
  return details => {
    try {
      onDenied?.(describeDeniedUrl(details.url))
    } catch {
      // observer failure is not a reason to reconsider the decision
    }

    return { action: 'deny' }
  }
}
