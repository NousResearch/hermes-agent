/**
 * Window translucency mapping shared by main-process call sites.
 *
 * Two modes ride one persisted lever (see src/store/translucency.ts):
 * - 'clear': the 0-100 intensity maps to native window opacity — the whole
 *   window (text included) fades so the desktop shows through unblurred.
 * - 'glass': the window stays fully opaque at the native level; the renderer
 *   thins its background surfaces instead so the macOS vibrancy material
 *   (already attached to every chat window) shows through as a matte blur
 *   while text keeps full contrast. macOS only — other platforms fall back
 *   to 'clear'.
 */

export type TranslucencyMode = 'clear' | 'glass'

export function clampIntensity(value: unknown): number {
  const n = Math.round(Number(value))

  return Number.isFinite(n) ? Math.min(100, Math.max(0, n)) : 0
}

/** Unknown or unsupported values fall back to 'clear' (the legacy behavior). */
export function normalizeMode(value: unknown, isMac: boolean): TranslucencyMode {
  return value === 'glass' && isMac ? 'glass' : 'clear'
}

/**
 * Native window opacity for a mode + intensity. Glass never fades the native
 * window — the see-through effect is painted by the renderer over vibrancy.
 * Clear keeps the historical ramp: floor at 0.3 so the most see-through
 * setting is still usable rather than nearly invisible. 0 → fully opaque.
 */
export function windowOpacityFor(intensity: number, mode: TranslucencyMode): number {
  if (mode === 'glass') {
    return 1
  }

  return 1 - (clampIntensity(intensity) / 100) * 0.7
}

/** Parse a persisted translucency.json / IPC payload into a safe shape. */
export function normalizePayload(payload: unknown, isMac: boolean): { intensity: number; mode: TranslucencyMode } {
  const record = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}

  return {
    intensity: clampIntensity(record.intensity),
    mode: normalizeMode(record.mode, isMac)
  }
}

/**
 * Whether glass is visually active. Decides the chat windows' webContents
 * backing: Chromium composites the page against the window's backgroundColor
 * BEFORE macOS composites the window, so an opaque backing (the normal
 * anti-white-flash paint) blocks the vibrancy material even under a fully
 * transparent page. Glass needs the backing gone; any other state keeps the
 * opaque themed backing.
 */
export function glassActive(state: { intensity: number; mode: TranslucencyMode }): boolean {
  return state.mode === 'glass' && state.intensity > 0
}

/**
 * BrowserWindow constructor options for a chat window's backing, given the
 * translucency state at creation time.
 *
 * Glass active → OMIT `backgroundColor` entirely: on a `vibrancy` window the
 * NSVisualEffectView then shows through a transparent page from the first
 * frame. Passing an alpha color instead does NOT work — the docs only support
 * constructor alpha with `transparent: true`, and `#00000000` on a normal
 * window is quietly treated as opaque.
 *
 * Glass inactive → the opaque themed backing (anti-flash paint before the
 * renderer's first paint, and what clear mode fades against).
 *
 * A runtime `setBackgroundColor` swap (see applyWindowTranslucency in main)
 * only settles reliably on a window that has been compositing for a while —
 * measured on macOS 26/Electron 40: swaps issued during roughly the first
 * seconds of a fresh process were lost, including from 'ready-to-show' and
 * 'did-finish-load' — so creation must not rely on a post-creation fixup.
 */
export function windowBackingOptions(
  state: { intensity: number; mode: TranslucencyMode },
  themedColor: string
): { backgroundColor?: string } {
  return glassActive(state) ? {} : { backgroundColor: themedColor }
}
