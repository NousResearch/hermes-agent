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

/**
 * macOS vibrancy materials offered as glass "frost" levels, ordered sheer →
 * heavy. macOS exposes no blur-radius knob (VibrancyOptions is only an
 * animation duration), so the material IS the frost control: each maps to a
 * different NSVisualEffectView material with its own luminance lift.
 *
 * Curated by pixel census on macOS 26 (one window, visualEffectState pinned
 * to 'active', cycling all 14 materials over the same wallpaper): the 14
 * collapse to 9 distinct looks (sidebar≡hud, window≡fullscreen-ui,
 * tooltip≡content≡under-window≡under-page). These four are the ladder with
 * the widest separations that stay distinct in BOTH appearances — dark lum
 * 26/63/84/127, light lum 217/233/254/242. sidebar/hud sit 9 lum from
 * under-window when focused and collapse INTO it when unfocused, which
 * shipped as two indistinguishable picker options once — don't re-add them.
 */
export const GLASS_MATERIALS = ['under-window', 'popover', 'titlebar', 'header'] as const

export type GlassMaterial = (typeof GLASS_MATERIALS)[number]

export const DEFAULT_GLASS_MATERIAL: GlassMaterial = 'under-window'

/**
 * Where the glass field lives. 'window' thins every field surface; 'sidebar'
 * is the Finder shape — glass rail, opaque content column. The scope is a
 * renderer concern (which surfaces thin); the main process only persists and
 * echoes it.
 */
export const GLASS_SCOPES = ['window', 'sidebar'] as const

export type GlassScope = (typeof GLASS_SCOPES)[number]

export const DEFAULT_GLASS_SCOPE: GlassScope = 'window'

export interface TranslucencyState {
  intensity: number
  mode: TranslucencyMode
  material: GlassMaterial
  scope: GlassScope
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
 * is the fallback everywhere else.
 *
 * With no mode recorded, macOS gets glass — it is the better-looking half of
 * the feature and the one worth finding, and pre-selecting it costs a fresh
 * profile nothing because the intensity still starts at 0 (the whole feature
 * is off until the user raises the lever). `legacyIntensity` is the escape
 * hatch: a profile that already carries a NON-ZERO intensity but no mode
 * predates this setting and has been rendering as clear all along, so it keeps
 * rendering as clear. Flipping a window someone already tuned is the one thing
 * a default must not do.
 */
export function normalizeMode(value: unknown, isMac: boolean, legacyIntensity = 0): TranslucencyMode {
  if (!isMac) {
    return 'clear'
  }

  if (value === 'glass' || value === 'clear') {
    return value
  }

  return legacyIntensity > 0 ? 'clear' : 'glass'
}

/** Unknown or unsupported values fall back to the default material. */
export function normalizeMaterial(value: unknown): GlassMaterial {
  return GLASS_MATERIALS.includes(value as GlassMaterial) ? (value as GlassMaterial) : DEFAULT_GLASS_MATERIAL
}

/** Unknown or unsupported values fall back to whole-window glass. */
export function normalizeScope(value: unknown): GlassScope {
  return GLASS_SCOPES.includes(value as GlassScope) ? (value as GlassScope) : DEFAULT_GLASS_SCOPE
}

/** Parse a persisted translucency.json / IPC payload into a safe state. */
export function normalizeState(payload: unknown, isMac: boolean): TranslucencyState {
  const record = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
  const intensity = clampIntensity(record.intensity)

  return {
    intensity,
    mode: normalizeMode(record.mode, isMac, intensity),
    material: normalizeMaterial(record.material),
    scope: normalizeScope(record.scope)
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
 * Percent of the surface tint the renderer KEEPS at a given intensity. Linear
 * to zero: at 100 the tint is fully gone — bare vibrancy glass — so the slider
 * spans the whole range from opaque theme to untinted blur. Text and cards
 * keep their own opaque tokens for contrast; only the field surfaces thin.
 */
export function glassSurfaceKeep(intensity: number): number {
  return TRANSLUCENCY_MAX - clampIntensity(intensity)
}

/**
 * The vibrancy material a chat window should carry. 'sidebar' is the
 * long-standing default the titlebar band was designed against; glass mode
 * swaps the whole window onto the user's chosen material (setVibrancy is
 * cheap and animatable at runtime, unlike the backing).
 */
export function vibrancyFor(state: TranslucencyState): GlassMaterial | 'sidebar' {
  return glassActive(state) ? state.material : 'sidebar'
}
