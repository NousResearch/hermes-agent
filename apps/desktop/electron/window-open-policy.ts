/**
 * Window-open policy for every BrowserWindow's webContents.
 *
 * In the Electron desktop app, every external URL we open on purpose is routed
 * through the audited `hermes:openExternal` IPC channel (see `openExternalUrl`
 * in main.ts, which enforces an http/https/mailto scheme allowlist and guards
 * file: through the IPC path resolver). The `window.open` / `target=_blank`
 * path that reaches `setWindowOpenHandler` is therefore, inside Electron, only
 * ever driven by content we did NOT initiate — most dangerously untrusted HTML
 * rendered in sandboxed `allow-scripts` iframes (artifact previews).
 *
 * GHSA-9f4c-93c8-jc8g (CVE-2026-70608, High 7.2) lets such a sandboxed iframe
 * reach this handler with NO user interaction and WITHOUT `allow-popups`. If
 * the handler opens `details.url` as a side effect, a malicious artifact can
 * force the user's real browser to an attacker-chosen URL. There is no fixed
 * 40.x Electron release (the fix is 41.10.3+/42.0.1), so we defend at the seam
 * regardless of Electron version: deny every window-open request and NEVER open
 * a URL as a side effect here. Trusted opens keep working because they go
 * through the IPC channel, not this handler.
 */

export interface WindowOpenRequestLike {
  url: string
}

export interface WindowOpenDecision {
  action: 'deny'
}

/**
 * The security decision for a window-open request. Always deny — see the module
 * comment. Kept as a named function so the contract has one tested home and a
 * future edit that tries to reintroduce conditional opening has to defeat the
 * test rather than silently succeed.
 */
export function decideWindowOpen(_request: WindowOpenRequestLike): WindowOpenDecision {
  return { action: 'deny' }
}

/**
 * Build a `setWindowOpenHandler` callback. It denies unconditionally and never
 * opens anything. `onDenied` is an optional observability hook (logging only);
 * it MUST NOT open a URL — doing so would re-open the CVE this handler closes.
 */
export function createWindowOpenHandler(
  onDenied?: (url: string) => void
): (details: WindowOpenRequestLike) => WindowOpenDecision {
  return details => {
    if (onDenied) {
      onDenied(details.url)
    }

    return decideWindowOpen(details)
  }
}
