/**
 * Window translucency — the one place the main process and the renderer agree
 * on what the setting means.
 *
 * One lever, 0–100 (0 = off, the default). Two modes decide HOW the desktop
 * shows through:
 *
 * - 'clear' — the main process maps the lever to native window opacity
 *   (`setOpacity`), so the whole window fades, text included. macOS + Windows;
 *   `setOpacity` is a no-op on Linux.
 * - 'glass' — macOS only. The window stays fully opaque at the native level and
 *   the renderer thins its page surfaces instead, letting the vibrancy material
 *   every chat window already carries read as a matte blur while text keeps
 *   full contrast.
 *
 * The renderer owns the value and mirrors it to main over IPC; main persists it
 * so a cold launch can apply it at window creation, before the renderer reports
 * anything.
 */

export type TranslucencyMode = 'clear' | 'glass'

export interface TranslucencyState {
  intensity: number
  mode: TranslucencyMode
}

export const TRANSLUCENCY_MIN = 0
export const TRANSLUCENCY_MAX = 100

/** Renderer slider granularity. Main accepts any integer in range. */
export const TRANSLUCENCY_STEP = 1

/** Most see-through clear setting — floored so it stays usable, not invisible. */
export const TRANSLUCENCY_OPACITY_FLOOR = 0.3

/**
 * Exponent for the clear intensity → opacity ramp. 1 is a linear ramp, which
 * spends the whole readable band (opacity ≳ 0.95) in the first few percent of
 * the lever. 2 holds that band across roughly the first third while leaving
 * both endpoints bit-identical to the linear ramp.
 */
export const TRANSLUCENCY_CURVE = 2

export function clampIntensity(value: unknown): number {
  const n = Math.round(Number(value))

  return Number.isFinite(n) ? Math.min(TRANSLUCENCY_MAX, Math.max(TRANSLUCENCY_MIN, n)) : TRANSLUCENCY_MIN
}

/**
 * Glass rides on the macOS vibrancy material, so it is macOS-only and 'clear'
 * is the fallback everywhere else. Clear is also the default: a profile with no
 * mode recorded predates this setting and always behaved as clear, so it keeps
 * behaving as clear rather than silently changing look on update.
 */
export function normalizeMode(value: unknown, isMac: boolean): TranslucencyMode {
  return value === 'glass' && isMac ? 'glass' : 'clear'
}

/** Parse a persisted translucency.json / IPC payload into a safe state. */
export function normalizeState(payload: unknown, isMac: boolean): TranslucencyState {
  const record = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}

  return {
    intensity: clampIntensity(record.intensity),
    mode: normalizeMode(record.mode, isMac)
  }
}

/**
 * Native window opacity for a state. Glass never fades the native window — its
 * see-through effect is painted by the renderer over the vibrancy material.
 */
export function windowOpacityFor({ intensity, mode }: TranslucencyState): number {
  if (mode === 'glass') {
    return 1
  }

  const ratio = clampIntensity(intensity) / TRANSLUCENCY_MAX

  return 1 - (1 - TRANSLUCENCY_OPACITY_FLOOR) * Math.pow(ratio, TRANSLUCENCY_CURVE)
}

/**
 * Whether glass is visually active. Both processes branch on this: main to
 * decide a window's backing, the renderer to decide whether to thin surfaces.
 */
export function glassActive({ intensity, mode }: TranslucencyState): boolean {
  return mode === 'glass' && intensity > 0
}

/**
 * Percent of the surface tint the renderer KEEPS at a given intensity. Mirrors
 * clear mode's opacity floor: at 100 the field still holds a 30% wash, so glass
 * reads as tinted blur rather than bare desktop.
 */
export function glassSurfaceKeep(intensity: number): number {
  return TRANSLUCENCY_MAX - clampIntensity(intensity) * (1 - TRANSLUCENCY_OPACITY_FLOOR)
}
