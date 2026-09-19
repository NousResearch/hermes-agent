/**
 * UI Scale presets, as zoom percentages. Kept on the main-process side next to
 * electron/zoom.ts — the module that owns the percent <-> zoom-level clamp —
 * so the range the Appearance row advertises can never drift from what the
 * engine actually honors (the same single-source pattern pool-limits uses,
 * review note on #92581). Every entry must stay inside the level clamp
 * (±9 levels, ≈30%–516%); zoom.test.ts pins the roundtrip invariant.
 */

export const UI_SCALE_PRESETS = ['90', '100', '110', '125', '150', '175', '200', '250'] as const

export type UiScalePreset = (typeof UI_SCALE_PRESETS)[number]

/** The UI Scale row highlights a preset only on an exact percent match; a
 *  Cmd/Ctrl step landing between presets intentionally highlights nothing. */
export function matchUiScalePreset(percent: number): UiScalePreset | null {
  return UI_SCALE_PRESETS.find(preset => Number(preset) === percent) ?? null
}
