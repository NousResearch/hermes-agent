'use strict'

// Classification + policy for the in-app browser's guest <webview> — kept pure
// and Electron-free so it's unit-testable. main.cjs performs the side effects
// (creating the popup BrowserWindow, handing off to the OS browser).
//
// WHY THIS EXISTS: the in-app browser (`browser-pane.tsx`, partition
// `persist:hermes-browser`) had NO main-process handler for guest popups. Many
// real sign-in flows — Google's "Sign in", most OAuth consent screens — drive a
// `window.open()` popup that must (a) share the opener's cookies/session and
// (b) be a genuine window so it can `postMessage` its result back and
// `window.close()` itself when done. With no `setWindowOpenHandler` on the
// guest, those popups were dropped or spawned uncontrolled, so sign-in hung.
//
// NOTE ON PASSKEYS: a popup window still cannot satisfy a WebAuthn *passkey*
// (Touch ID) challenge — Electron's bundled Chromium has no macOS
// platform-authenticator bridge, and an unsigned build is blocked from the
// secure enclave regardless. The popup fix restores password / emailed-code /
// "tap your phone" verification (the flows that were silently broken); the
// "Open in default browser" toolbar button is the escape hatch for the
// passkey-only case.

const IN_APP_BROWSER_PARTITION = 'persist:hermes-browser'

// Reasonable default chrome for an OAuth/sign-in popup window. Sized like a
// real browser auth popup; the host applies the shared in-app browser session
// so cookies flow and the popup can talk back to its opener.
const GUEST_POPUP_WINDOW = Object.freeze({ width: 520, height: 640 })

/**
 * Decide what to do with a `window.open()` / `target=_blank` originating in the
 * in-app browser guest. Pure — returns a verdict; the caller does the work.
 *
 *   { action: 'popup' }    → open as a child window in the SAME session so
 *                            OAuth/sign-in flows share cookies and can
 *                            postMessage/close back to the opener.
 *   { action: 'external' } → hand the URL to the OS default browser (non-web
 *                            schemes: mailto:, tel:, etc.). openExternalUrl on
 *                            the host re-validates and rejects unsafe schemes
 *                            (javascript:, data:, …), so this stays safe.
 *   { action: 'deny' }     → drop (empty / non-string URL).
 *
 * `about:blank` is intentionally a popup: a common OAuth pattern opens a blank
 * popup first, then navigates it to the provider — denying it breaks the flow.
 */
function classifyGuestWindowOpen(url) {
  if (typeof url !== 'string' || !url.trim()) {
    return { action: 'deny' }
  }

  const raw = url.trim()

  if (/^https?:\/\//i.test(raw) || /^about:blank/i.test(raw)) {
    return { action: 'popup' }
  }

  return { action: 'external' }
}

module.exports = {
  GUEST_POPUP_WINDOW,
  IN_APP_BROWSER_PARTITION,
  classifyGuestWindowOpen
}
