/**
 * Accent retinting — re-seed a theme's accent family from one color.
 *
 * A palette's accent is not one color, it's a family: the seed plus the soft
 * surfaces mixed from it. In `nous` that's 7 slots per mode —
 * `primary`/`ring`/`midground`/`composerRing` carry the accent itself, and
 * `accent`/`secondary`/`userBubble` are mixes of it toward the background. The
 * carrying slots need not be the same hex: a theme may shade them (see
 * `tracksAccent`), and a re-seed keeps that shading.
 *
 * So retinting is not "replace some hexes": it's move the seed, then re-derive
 * the family with the exact ratios the VS Code converter used. Those ratios
 * live in ACCENT_MIX below and are duplicated from `vscode.ts` on purpose —
 * they're verified to reproduce the shipped palette bit-for-bit (see
 * retint.test.ts), which is what lets a retint with the original seed be a
 * no-op.
 *
 * ONE color in, both modes out. A single hex can't serve both appearances
 * literally — Nous blue `#0053FD` is 5.4:1 on GitHub's light sidebar but only
 * 3.6:1 on its dark one, i.e. unreadable section headers. Dark therefore gets
 * the same hue and chroma at whatever lightness clears AA, which keeps the
 * brand color recognizably itself instead of washing it toward white.
 *
 * The chrome is deliberately untouched. GitHub's neutrals are a cool blue-gray
 * (hue ~210°) that reads as the app's surface, not as brand — rotating it with
 * the accent turns every theme into a monochrome wash.
 */

import {
  ensureContrastOklch,
  hexToOklch,
  hueDelta,
  mixOklab,
  normalizeHex,
  type Oklch,
  oklchToHex,
  readableOn
} from './color'
import type { DesktopTheme, DesktopThemeColors } from './types'

/** Small uppercase sidebar labels paint in the accent, so it must clear AA. */
const ACCENT_MIN_CONTRAST = 4.5

/**
 * How far a slot's hue may sit from the primary and still count as the same
 * accent family. Wide enough for a theme that shades its ring (midnight runs
 * 286° against a 293° primary), narrow enough that a second, deliberately
 * different accent is left alone.
 */
const ACCENT_HUE_TOLERANCE = 25

/** Below this chroma a colour is a grey and its hue is noise. */
const ACCENT_MIN_CHROMA = 0.03

/** Slots that carry the accent itself, rather than a surface mixed from it. */
const ACCENT_SEED_KEYS = ['primary', 'ring', 'midground', 'composerRing'] as const

/**
 * Slots mixed from the seed, as ratios per mode — the converter's own numbers.
 * `userBubble` mixes the CARD toward the accent (not the reverse).
 */
const ACCENT_MIX = {
  accent: { light: 0.88, dark: 0.82 },
  secondary: { light: 0.86, dark: 0.72 },
  userBubble: { light: 0.12, dark: 0.18 }
} as const

/**
 * Whether a slot belongs to the accent family, and so should follow a re-seed.
 *
 * Exact equality with `primary` is too strict: midnight's ring is `#8b80e8`
 * against a `#ddd6ff` primary — a deeper shade of the same violet, obviously
 * accent, and leaving it behind half-retints the theme. Hue is the honest
 * test. A slot within ACCENT_HUE_TOLERANCE of the primary is the same colour
 * family at another lightness; anything further is a deliberate choice (mono's
 * neutral grey ring) and stays put.
 *
 * Near-greys are decided by chroma, not hue: below ACCENT_MIN_CHROMA a hue
 * reading is noise, so a grey tracks the accent only when the accent is itself
 * a grey — which is how mono's own primary keeps working.
 */
function tracksAccent(slot: Oklch, primary: Oklch): boolean {
  if (slot.c < ACCENT_MIN_CHROMA) {
    return primary.c < ACCENT_MIN_CHROMA
  }

  return Math.abs(hueDelta(primary.h, slot.h)) <= ACCENT_HUE_TOLERANCE
}

/** Re-derive the accent family of one palette around `seed`. */
function retintColors(colors: DesktopThemeColors, seed: string, isDark: boolean): DesktopThemeColors {
  const next: DesktopThemeColors = { ...colors }
  const pick = (spec: { dark: number; light: number }) => (isDark ? spec.dark : spec.light)
  const primary = hexToOklch(colors.primary)
  const seedLch = hexToOklch(seed)

  for (const key of ACCENT_SEED_KEYS) {
    const slot = colors[key]
    const lch = slot ? hexToOklch(slot) : null

    if (!lch || !primary || !seedLch || !tracksAccent(lch, primary)) {
      continue
    }

    // Take the seed's hue but keep the slot's OWN lightness and chroma, so a
    // theme that deliberately runs a deeper ring than its primary keeps that
    // relationship instead of being flattened onto one colour.
    next[key] = slot === colors.primary ? seed : oklchToHex({ ...lch, h: seedLch.h })
  }

  // OKLab, not sRGB: blending a saturated blue toward white in gamma space
  // drifts it ~8° violet, which is exactly how a clean blue accent produced a
  // lavender selection row. See mixOklab.
  next.accent = mixOklab(seed, colors.background, pick(ACCENT_MIX.accent))
  next.secondary = mixOklab(seed, colors.background, pick(ACCENT_MIX.secondary))
  next.userBubble = mixOklab(colors.card, seed, pick(ACCENT_MIX.userBubble))

  // Foregrounds that sit ON the accent have to be re-picked: a new seed can
  // cross the light/dark readability boundary.
  next.primaryForeground = readableOn(seed)
  next.midgroundForeground = readableOn(seed)

  return next
}

/** The seed a palette should use, adapted to stay legible on its own sidebar. */
function seedFor(colors: DesktopThemeColors, seed: string): string {
  return ensureContrastOklch(seed, colors.sidebarBackground ?? colors.background, ACCENT_MIN_CONTRAST)
}

/**
 * Carry a light-mode seed across to dark by the theme's OWN lightness offset.
 *
 * A theme that ships both palettes has already answered "how much lighter does
 * this accent get in dark mode" — nous's blues are OKLCH L 0.528 → 0.638 at the
 * same hue. Reapplying that delta means a picked color lands in dark exactly
 * where the author would have put it, and it makes the round trip exact:
 * retinting with the theme's own accent reproduces both palettes byte-for-byte.
 *
 * Falling back to a bare AA clamp (what this did first) technically passed
 * contrast but ignored the author's intent — GitHub green came back as
 * `#368548` instead of `#4f9e5e`, a visibly duller dark accent.
 */
function carryToDark(theme: DesktopTheme, seed: string): string {
  const light = hexToOklch(theme.colors.primary)
  const dark = theme.darkColors ? hexToOklch(theme.darkColors.primary) : null
  const picked = hexToOklch(seed)

  if (!light || !dark || !picked) {
    return seed
  }

  const shifted = oklchToHex({ ...picked, l: Math.min(1, Math.max(0, picked.l + (dark.l - light.l))) })

  return shifted
}

/**
 * Return `theme` with its accent family re-seeded from `color` (any CSS hex).
 *
 * Each mode gets the seed adapted to its own surface, so one pick stays
 * readable in both. Passing a theme's existing accent returns it untouched.
 * An unparseable color returns the theme unchanged rather than throwing —
 * callers are often wiring this straight to a text input.
 */
export function retintTheme(theme: DesktopTheme, color: string): DesktopTheme {
  const seed = normalizeHex(color)

  if (!seed) {
    return theme
  }

  // Re-seeding with the theme's own accent must return the theme untouched.
  // Recomputing dark from it would round-trip through OKLCH and land a bit or
  // two off (#4f9e5e → #509e5f) — invisible, but it would mean "no change"
  // wasn't actually no change.
  if (seed.toLowerCase() === theme.colors.primary.toLowerCase()) {
    return theme
  }

  const light = seedFor(theme.colors, seed)

  const retinted: DesktopTheme = {
    ...theme,
    colors: theme.colors.primary === light ? theme.colors : retintColors(theme.colors, light, false)
  }

  if (theme.darkColors) {
    // Carry by the theme's own light→dark offset first, THEN clamp — so the
    // author's intent leads and contrast is only a floor, never the target.
    const dark = seedFor(theme.darkColors, carryToDark(theme, seed))

    retinted.darkColors =
      theme.darkColors.primary === dark ? theme.darkColors : retintColors(theme.darkColors, dark, true)
  }

  return retinted
}

/** The theme's current accent hue in degrees. */
export function themeHue(theme: DesktopTheme): number {
  const lch = hexToOklch(theme.colors.primary)

  return lch ? Math.round(lch.h) : 0
}
