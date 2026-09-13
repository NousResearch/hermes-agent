/**
 * Platform detection for the renderer.
 *
 * The renderer has no `process.platform`, and several surfaces need to know
 * which OS they're on — keybind glyphs, terminal shortcuts, and glass. One
 * definition so they can't disagree.
 *
 * Win10 vs Win11 is not visible here (both report NT 10.0). The real glass
 * gate is `hermesDesktop.glassSupported`, which main/preload compute from
 * `os.release()`.
 */

export const isMacPlatform = (): boolean =>
  typeof navigator !== 'undefined' && /mac/i.test(navigator.platform || navigator.userAgent || '')

// Not `/win/i` — that matches the substring inside `darwin`, which is jsdom's
// default userAgent. Win32 / Windows NT are the real tokens.
export const isWindowsPlatform = (): boolean =>
  typeof navigator !== 'undefined' && /win32|windows/i.test(navigator.platform || navigator.userAgent || '')

export const isLinuxPlatform = (): boolean =>
  typeof navigator !== 'undefined' && /linux/i.test(navigator.platform || navigator.userAgent || '')

/** Renderer host, independent of the OS or the gateway's connection mode. */
export function isBrowserHostedDesktop() {
  if (typeof document === 'undefined') {
    return false
  }

  // The bridge writes the marker once it installs. During static module
  // evaluation, use the browser bootstrap globals too; otherwise an OS check
  // can classify a browser page as native before the marker is written.
  const win = window as Window & {
    __HERMES_AUTH_REQUIRED__?: boolean
    __HERMES_SESSION_TOKEN__?: string
  }

  return document.documentElement.dataset.hermesDesktopHost === 'browser' ||
    (!window.hermesDesktop && (win.__HERMES_AUTH_REQUIRED__ === true || Boolean(win.__HERMES_SESSION_TOKEN__)))
}
