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
