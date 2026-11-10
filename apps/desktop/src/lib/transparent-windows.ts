// The `?win=` kinds whose Electron window is `transparent: true` and so paints
// nothing but its own surface over the user's desktop. `secondary` (a session
// window) and `browser` are ordinary opaque windows and are deliberately not
// in here. index.html's pre-paint script skips exactly this list — keep the
// two in step. No imports: the renderer entry reads it before the app loads.
export const TRANSPARENT_WINDOWS = new Set(['hud', 'overlay', 'quick', 'wake'])

export function isTransparentWindow(search = window.location.search): boolean {
  const win = new URLSearchParams(search).get('win')

  return Boolean(win && TRANSPARENT_WINDOWS.has(win))
}
