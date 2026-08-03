/**
 * Pure helpers for window translucency. The main process owns
 * BrowserWindow#setOpacity, so both window creation and the runtime IPC toggle
 * funnel through this one mapping. Intensity is the user-facing unit
 * (0 = opaque, 100 = most see-through); Electron's unit is window opacity.
 *
 * `setOpacity` fades the whole window, text included, so the band a user can
 * actually read in sits just under opacity 1. A linear ramp spends that band in
 * its first few percent and the rest of the lever on settings nobody can use.
 * Curving the ramp keeps both endpoints exactly where they have always been and
 * gives the readable band most of the travel instead.
 */

export const TRANSLUCENCY_MIN = 0
export const TRANSLUCENCY_MAX = 100

/** Most see-through setting — floored so it stays usable, not near-invisible. */
export const TRANSLUCENCY_OPACITY_FLOOR = 0.3

/**
 * Exponent for the intensity -> opacity ramp. 1 is the linear ramp this
 * replaces. 2 holds the readable settings across roughly the first third of the
 * lever while leaving windowOpacity(0) and windowOpacity(100) bit-identical.
 */
export const TRANSLUCENCY_CURVE = 2

export function clampIntensity(value) {
  const n = Math.round(Number(value))

  return Number.isFinite(n) ? Math.min(TRANSLUCENCY_MAX, Math.max(TRANSLUCENCY_MIN, n)) : TRANSLUCENCY_MIN
}

export function windowOpacity(intensity) {
  const ratio = clampIntensity(intensity) / TRANSLUCENCY_MAX

  return 1 - (1 - TRANSLUCENCY_OPACITY_FLOOR) * Math.pow(ratio, TRANSLUCENCY_CURVE)
}
