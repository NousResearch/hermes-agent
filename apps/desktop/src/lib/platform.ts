/** Best-effort OS detection in the renderer (no node APIs). */

export const isWindows = (): boolean =>
  typeof navigator !== 'undefined' && /win/i.test(navigator.platform || navigator.userAgent)

export const isMac = (): boolean =>
  typeof navigator !== 'undefined' && /mac/i.test(navigator.platform || navigator.userAgent)
