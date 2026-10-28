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

/** Globals the Hermes web server injects into the page it serves. */
export interface BrowserHostGlobals {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_BASE_PATH__?: string
  __HERMES_DASHBOARD_PROFILE__?: string
  __HERMES_UI_SURFACE__?: string
}

/**
 * The profile the server that served this page runs as: its gateway socket and its REST answer for
 * this profile when a request names none. Empty when the server cannot name its own home (a custom
 * HERMES_HOME), which the browser host reads as `default`.
 */
export function browserLaunchProfile(): string {
  return String((window as Window & BrowserHostGlobals).__HERMES_DASHBOARD_PROFILE__ ?? '').trim()
}

/** The profile a request that names none reaches: the one the server runs as, else `default`. */
export function browserLaunchProfileOrDefault(): string {
  return browserLaunchProfile() || 'default'
}

/**
 * The server served this page as the browser-hosted Desktop: a gated host
 * (cookie auth) or the Webapp surface (its session arrives by launch link).
 * The browser bridge installs on exactly this predicate. An injected session
 * token alone is not one: nothing serves this renderer that way, and the only
 * gateway route it could use is the Webapp's.
 */
export function hasBrowserHostBootstrap(): boolean {
  const win = window as Window & BrowserHostGlobals

  return win.__HERMES_AUTH_REQUIRED__ === true || win.__HERMES_UI_SURFACE__ === 'webapp'
}

/** Renderer host, independent of the OS or the gateway's connection mode. */
export function isBrowserHostedDesktop() {
  if (typeof document === 'undefined') {
    return false
  }

  // The bridge writes the marker once it installs. During static module
  // evaluation, use the browser bootstrap globals too; otherwise an OS check
  // can classify a browser page as native before the marker is written.
  return (
    document.documentElement.dataset.hermesDesktopHost === 'browser' ||
    (!window.hermesDesktop && hasBrowserHostBootstrap())
  )
}

/**
 * The profile the renderer's gateway route starts on, before boot adopts its primary. A browser
 * host's boot adopts the profile its server runs as, and requests that name the tab's profile can
 * go out before that: on `default` they would name a profile a gateway launched as another one may
 * not serve, and it refuses them (`profile_mismatch`). Native Desktop starts on `default`.
 */
export function initialGatewayProfile(): string {
  return isBrowserHostedDesktop() ? browserLaunchProfileOrDefault() : 'default'
}
