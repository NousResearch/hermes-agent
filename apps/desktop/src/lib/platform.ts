/**
 * Platform detection for the renderer.
 *
 * The renderer has no `process.platform`, and several surfaces need to know
 * whether they're on a Mac — keybind glyphs, terminal shortcuts, and the
 * macOS-only vibrancy features. One definition so they can't disagree.
 */

export const isMacPlatform = (): boolean =>
  typeof navigator !== 'undefined' && /mac/i.test(navigator.platform || navigator.userAgent || '')
