/**
 * THE sRGB color primitive for every TypeScript surface (desktop themes, the
 * Ink TUI). Parse, mix, measure and fix colors here so tone ladders, contrast
 * floors and one-off UI needs share one set of semantics — and so the two
 * surfaces can never drift apart on what "readable" means again.
 *
 * Mixing is an sRGB lerp — deliberately, for byte parity with the desktop's
 * `color-mix(in srgb, ...)` ladder in styles.css. Perceptual (OKLCH) math is
 * desktop-only and lives in `apps/desktop/src/themes/color.ts`.
 *
 * Everything that measures returns `null` for unparseable input rather than 0:
 * a silent 0 makes `ansi256(245)` look like pure black and passes garbage
 * through contrast checks as if it were a real measurement.
 */

export type Rgb = readonly [number, number, number]

const HEX6_RE = /^#?([0-9a-f]{6})$/i
const HEX3_RE = /^#?([0-9a-f]{3})$/i
const RGB_FN_RE = /^rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})/i

const clampChannel = (v: number) => Math.max(0, Math.min(255, Math.round(v)))

/** Parse `#rgb`, `#rrggbb` or `rgb(r,g,b)` → channels. Null for anything else. */
export function parseColor(input: string): Rgb | null {
  const value = input.trim()

  let m = HEX6_RE.exec(value)

  if (m) {
    const n = parseInt(m[1]!, 16)

    return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]
  }

  m = HEX3_RE.exec(value)

  if (m) {
    const [r, g, b] = m[1]!

    return [parseInt(r! + r!, 16), parseInt(g! + g!, 16), parseInt(b! + b!, 16)]
  }

  m = RGB_FN_RE.exec(value)

  if (m) {
    return [clampChannel(Number(m[1])), clampChannel(Number(m[2])), clampChannel(Number(m[3]))]
  }

  return null
}

export const toHex = (rgb: Rgb): string => '#' + rgb.map(c => clampChannel(c).toString(16).padStart(2, '0')).join('')

/** sRGB lerp `a → b` by `t` in [0,1]. Unparseable inputs return `a` unchanged. */
export function mix(a: string, b: string, t: number): string {
  const pa = parseColor(a)
  const pb = parseColor(b)

  if (!pa || !pb) {
    return a
  }

  return toHex([pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t, pa[2] + (pb[2] - pa[2]) * t])
}

function channelLuminance(value: number): number {
  const normalized = value / 255

  return normalized <= 0.03928 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4
}

/** WCAG relative luminance in [0,1]. Null when unparseable. */
export function relativeLuminance(color: string): null | number {
  const rgb = parseColor(color)

  return rgb
    ? 0.2126 * channelLuminance(rgb[0]) + 0.7152 * channelLuminance(rgb[1]) + 0.0722 * channelLuminance(rgb[2])
    : null
}

/** WCAG contrast ratio between two colors (1–21). Null when unparseable. */
export function contrastRatio(a: string, b: string): null | number {
  const la = relativeLuminance(a)
  const lb = relativeLuminance(b)

  if (la === null || lb === null) {
    return null
  }

  const [hi, lo] = la >= lb ? [la, lb] : [lb, la]

  return (hi + 0.05) / (lo + 0.05)
}

const DEFAULT_INKS = ['#000000', '#ffffff'] as const

/**
 * The readable ink for a background: whichever candidate MEASURES better.
 *
 * Splitting on a luminance threshold got mid-lightness accents wrong in the
 * direction that matters — white on GitHub's dark green `#4f9e5e` is 3.29:1
 * (fails AA) where near-black is 5.50:1, and Catppuccin's mauve was a 2.03:1
 * white-on-lilac. Two candidates is a cheap enough search to just measure.
 * Ties and an unparseable background (nothing to measure against) go to the
 * LAST candidate — the light pole in both the default pair and the desktop's,
 * which is what both surfaces previously fell back to.
 */
export function readableOn(bg: string, inks: readonly [string, ...string[]] = DEFAULT_INKS): string {
  let best: string = inks[inks.length - 1]!
  let bestRatio = -1

  for (const ink of inks) {
    const ratio = contrastRatio(bg, ink)

    if (ratio !== null && ratio >= bestRatio) {
      best = ink
      bestRatio = ratio
    }
  }

  return best
}

/**
 * Step-mix `color` toward the readable pole of `bg` until the contrast ratio
 * clears `min`. Each step re-mixes from the ORIGINAL color in 0.05 increments
 * so hue decays linearly, not exponentially, and the color stops at the first
 * rung that passes — minimal hue loss for the ratio. Returns the original when
 * it already passes or isn't parseable.
 */
export function ensureContrast(color: string, bg: string, min: number): string {
  if (relativeLuminance(bg) === null || parseColor(color) === null) {
    return color
  }

  const pole = readableOn(bg)
  let current = color

  for (let step = 0; step <= 20; step++) {
    const ratio = contrastRatio(current, bg)

    if (ratio === null || ratio >= min) {
      return current
    }

    current = mix(color, pole, Math.min(1, (step + 1) * 0.05))
  }

  return current
}

/** Recede toward the background pole (opposite of `readableOn`). */
export const lighten = (color: string, t: number) => mix(color, '#ffffff', t)
export const darken = (color: string, t: number) => mix(color, '#000000', t)
