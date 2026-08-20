/**
 * Small color helpers shared by the theme context (synthesised light variants)
 * and the VS Code theme converter (token → seed mapping).
 *
 * Everything works in 6-digit `#rrggbb`. `normalizeHex` is the front door for
 * untrusted input (VS Code themes use `#rgb`, `#rgba`, `#rrggbbaa`, and named
 * tokens), flattening alpha over a backdrop so downstream math stays simple.
 */

export function hexToRgb(hex: string): [number, number, number] | null {
  const clean = hex.trim().replace(/^#/, '')

  if (!/^[0-9a-f]{6}$/i.test(clean)) {
    return null
  }

  return [0, 2, 4].map(i => parseInt(clean.slice(i, i + 2), 16)) as [number, number, number]
}

export const rgbToHex = ([r, g, b]: [number, number, number]): string =>
  `#${[r, g, b]
    .map(n =>
      Math.round(Math.min(255, Math.max(0, n)))
        .toString(16)
        .padStart(2, '0')
    )
    .join('')}`

export function mix(a: string, b: string, amount: number): string {
  const ar = hexToRgb(a)
  const br = hexToRgb(b)

  return ar && br
    ? rgbToHex([ar[0] + (br[0] - ar[0]) * amount, ar[1] + (br[1] - ar[1]) * amount, ar[2] + (br[2] - ar[2]) * amount])
    : a
}

const linearize = (channel: number): number =>
  channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4

/** WCAG relative luminance (gamma-corrected), 0..1. */
export function relativeLuminance(hex: string): number {
  const rgb = hexToRgb(hex)

  if (!rgb) {
    return 0
  }

  const [r, g, b] = rgb.map(v => linearize(v / 255))

  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

/** WCAG contrast ratio (1..21) between two hex colors. */
export function contrastRatio(a: string, b: string): number {
  const la = relativeLuminance(a)
  const lb = relativeLuminance(b)

  return la >= lb ? (la + 0.05) / (lb + 0.05) : (lb + 0.05) / (la + 0.05)
}

/**
 * A readable foreground (`#161616` or `#ffffff`) for a background hex.
 *
 * Picks whichever of the two actually measures better, rather than splitting on
 * a luminance threshold. The threshold version got mid-lightness accents wrong
 * in the direction that matters — white on GitHub's dark green `#4f9e5e` is
 * 3.29:1 (fails AA) where near-black is 5.50:1, and Catppuccin's mauve was a
 * 2.03:1 white-on-lilac. Two candidates is a cheap enough search to just
 * measure.
 */
export function readableOn(hex: string): string {
  return contrastRatio(hex, '#ffffff') >= contrastRatio(hex, '#161616') ? '#ffffff' : '#161616'
}

/**
 * Guarantee `color` reads against `bg`: if it's below `min` contrast, mix it
 * toward white (on a dark bg) or black (on a light bg) in steps until it clears,
 * keeping the hue as much as possible. Used so imported accents never collapse
 * into a near-background sidebar (the "invisible label" case).
 */
export function ensureContrast(color: string, bg: string, min: number): string {
  if (contrastRatio(color, bg) >= min) {
    return color
  }

  const towards = relativeLuminance(bg) < 0.5 ? '#ffffff' : '#000000'
  let best = color

  for (let amount = 0.2; amount <= 1.0001; amount += 0.2) {
    best = mix(color, towards, Math.min(amount, 1))

    if (contrastRatio(best, bg) >= min) {
      return best
    }
  }

  return best
}

/** Perceptual-ish luminance in 0..1 (naive, for light/dark bucketing). */
export function luminance(hex: string): number {
  const rgb = hexToRgb(hex)

  if (!rgb) {
    return 0
  }

  const [r, g, b] = rgb.map(v => v / 255)

  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

/**
 * Coerce any CSS hex color VS Code themes throw at us into a flat 6-digit
 * `#rrggbb`, compositing alpha over `backdrop`. Accepts `#rgb`, `#rgba`,
 * `#rrggbb`, `#rrggbbaa` (with or without the leading `#`). Returns null for
 * non-hex values (named colors, `rgb()`, etc.) so callers can fall back.
 */
export function normalizeHex(input: string | undefined | null, backdrop = '#000000'): string | null {
  if (typeof input !== 'string') {
    return null
  }

  let clean = input.trim().replace(/^#/, '')

  // Expand shorthand (#rgb / #rgba) to full width.
  if (clean.length === 3 || clean.length === 4) {
    clean = clean
      .split('')
      .map(ch => ch + ch)
      .join('')
  }

  if (!/^[0-9a-f]{6}([0-9a-f]{2})?$/i.test(clean)) {
    return null
  }

  const rgb = hexToRgb(`#${clean.slice(0, 6)}`)

  if (!rgb) {
    return null
  }

  if (clean.length === 6) {
    return rgbToHex(rgb)
  }

  const alpha = parseInt(clean.slice(6, 8), 16) / 255
  const base = hexToRgb(backdrop) ?? [0, 0, 0]

  return rgbToHex([
    base[0] + (rgb[0] - base[0]) * alpha,
    base[1] + (rgb[1] - base[1]) * alpha,
    base[2] + (rgb[2] - base[2]) * alpha
  ])
}

// ─── OKLCH ──────────────────────────────────────────────────────────────────
// Hue rotation has to happen in a perceptual space or it lies. Holding HSL's
// saturation+lightness across a hue sweep gives you mud at 60° (`#6d6d19`) and
// a washed teal at 200°, because HSL "lightness" is not lightness. OKLCH holds
// perceived lightness and colorfulness steady, so every hue lands with the same
// visual weight — which is what makes a single hue knob safe to expose.

const linearize01 = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
const delinearize01 = (c: number): number => (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055)

/** Perceptual lightness 0..1, chroma (~0..0.37), hue 0..360. */
export interface Oklch {
  l: number
  c: number
  h: number
}

export function hexToOklch(hex: string): Oklch | null {
  const rgb = hexToRgb(hex)

  if (!rgb) {
    return null
  }

  const [okL, okA, okB] = rgbToOklab(rgb)

  return {
    l: okL,
    c: Math.hypot(okA, okB),
    h: ((Math.atan2(okB, okA) * 180) / Math.PI + 360) % 360
  }
}

/** 0–255 sRGB triple → OKLab [L, a, b]. */
function rgbToOklab([r255, g255, b255]: readonly number[]): [number, number, number] {
  const [r, g, b] = [r255, g255, b255].map(v => linearize01(v / 255))
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)

  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s
  ]
}

/** OKLab [L, a, b] → `#rrggbb`, via the chroma-preserving gamut fit. */
function oklabToHex([okL, okA, okB]: readonly number[]): string {
  return oklchToHex({
    l: okL,
    c: Math.hypot(okA, okB),
    h: ((Math.atan2(okB, okA) * 180) / Math.PI + 360) % 360
  })
}

/** Raw (possibly out-of-gamut) linear-sRGB triple for an OKLCH color. */
function oklchToRgbRaw({ l: okL, c, h }: Oklch): [number, number, number] {
  const rad = (h * Math.PI) / 180
  const okA = c * Math.cos(rad)
  const okB = c * Math.sin(rad)

  const l = (okL + 0.3963377774 * okA + 0.2158037573 * okB) ** 3
  const m = (okL - 0.1055613458 * okA - 0.0638541728 * okB) ** 3
  const s = (okL - 0.0894841775 * okA - 1.291485548 * okB) ** 3

  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s
  ]
}

const inSrgbGamut = (rgb: readonly number[]): boolean => rgb.every(c => c >= -0.001 && c <= 1.001)

/**
 * OKLCH → `#rrggbb`, reducing chroma until the color fits sRGB.
 *
 * Clipping the channels instead would shift hue and lightness — a saturated
 * blue clips to a different blue. Binary-searching chroma keeps the hue and the
 * perceived lightness exactly, and only gives up the colorfulness the display
 * cannot show. Deep blues and yellows need this; mid-chroma greens never do.
 */
export function oklchToHex(color: Oklch): string {
  if (inSrgbGamut(oklchToRgbRaw(color))) {
    return rgbToHex(oklchToRgbRaw(color).map(c => delinearize01(c) * 255) as [number, number, number])
  }

  let lo = 0
  let hi = color.c

  for (let i = 0; i < 24; i += 1) {
    const mid = (lo + hi) / 2

    if (inSrgbGamut(oklchToRgbRaw({ ...color, c: mid }))) {
      lo = mid
    } else {
      hi = mid
    }
  }

  return rgbToHex(oklchToRgbRaw({ ...color, c: lo }).map(c => delinearize01(c) * 255) as [number, number, number])
}

/**
 * OKLCH → 0–255 sRGB, or `null` when the color is outside the display's gamut.
 *
 * The picker draws its field with this: a hue slice of OKLCH is a curved wedge
 * in sRGB, not a rectangle, so plotting lightness against chroma needs to know
 * where the wedge ENDS. Rendering the boundary honestly is the difference
 * between a picker where every pixel is a real color and one where a third of
 * the area silently clamps to the same few hexes.
 */
export function oklchToSrgb255(color: Oklch): [number, number, number] | null {
  const raw = oklchToRgbRaw(color)

  if (!inSrgbGamut(raw)) {
    return null
  }

  return raw.map(c => Math.round(Math.min(1, Math.max(0, delinearize01(c))) * 255)) as [number, number, number]
}

/** Largest in-gamut chroma for a lightness+hue, for scaling the picker's axis. */
export function maxChroma(l: number, h: number): number {
  let lo = 0
  let hi = 0.4

  for (let i = 0; i < 20; i += 1) {
    const mid = (lo + hi) / 2

    if (inSrgbGamut(oklchToRgbRaw({ l, c: mid, h }))) {
      lo = mid
    } else {
      hi = mid
    }
  }

  return lo
}

/**
 * Bend a semantic color toward the accent along the shortest hue arc.
 *
 * A success green next to a blue accent reads as a clash, but recoloring it to
 * the accent throws away the meaning — "done" and "running" become one color.
 * Rotating it PART of the way keeps the semantics and settles the palette: at
 * `strength` 0 nothing moves, at 1 it lands on the accent hue.
 *
 * The default costs nothing by construction: when the accent already is a green
 * (GitHub's 148° vs emerald's 162°) the arc is 14° and a 0.25 rotation moves the
 * dot ~3° — invisible. The work only happens when the accent is genuinely far
 * away, which is exactly the case that was clashing.
 */
export function harmonize(hex: string, accent: string, strength: number): string {
  const base = hexToOklch(hex)
  const target = hexToOklch(accent)

  if (!base || !target) {
    return hex
  }

  // Signed shortest arc, so a green never takes the long way round the wheel.
  const delta = ((target.h - base.h + 540) % 360) - 180
  const h = (base.h + delta * Math.min(1, Math.max(0, strength)) + 360) % 360

  return oklchToHex({ ...base, c: Math.min(Math.max(base.c, target.c * 0.85), maxChroma(base.l, h)), h })
}

/**
 * Blend two colors in OKLab — the hue-stable counterpart to `mix`.
 *
 * `mix` lerps gamma-encoded sRGB channels, which bends hue on the way: a
 * saturated blue mixed 88% toward white lands 7.6° violet of where it started,
 * which is why a clean blue accent produced a lavender selection row. OKLab is
 * (near) perceptually uniform, so the same blend keeps the hue and just walks
 * chroma and lightness — the soft surface reads as a pale version of the accent
 * instead of a different color.
 *
 * Use this for anything derived from the brand seed. `mix` is still correct for
 * blending neutrals, where there is no hue to preserve.
 */
export function mixOklab(a: string, b: string, amount: number): string {
  const ra = hexToRgb(a)
  const rb = hexToRgb(b)

  if (!ra || !rb) {
    return a
  }

  const la = rgbToOklab(ra)
  const lb = rgbToOklab(rb)
  const t = Math.min(1, Math.max(0, amount))

  return oklabToHex([la[0] + (lb[0] - la[0]) * t, la[1] + (lb[1] - la[1]) * t, la[2] + (lb[2] - la[2]) * t])
}

/** Re-hue a color, holding its perceived lightness and colorfulness. */
export function withHue(hex: string, hue: number): string {
  const lch = hexToOklch(hex)

  return lch ? oklchToHex({ ...lch, h: ((hue % 360) + 360) % 360 }) : hex
}

/**
 * Make `hex` clear `min` contrast against `bg` by moving its LIGHTNESS only.
 *
 * The sRGB-mixing version of this (`ensureContrast`) blends toward white or
 * black, which drags chroma down with it — Nous blue `#0053FD` mixed 30% white
 * to pass AA on a dark surface loses a third of its colorfulness and reads
 * washed. Walking OKLCH lightness instead keeps hue and chroma intact, so a
 * brand color stays recognizably itself at whatever lightness the surface
 * demands. Returns the original when it already passes, and gives up at the
 * ends of the lightness range rather than looping.
 */
export function ensureContrastOklch(hex: string, bg: string, min: number): string {
  if (contrastRatio(hex, bg) >= min) {
    return hex
  }

  const lch = hexToOklch(hex)

  if (!lch) {
    return hex
  }

  // Lighten on a dark backdrop, darken on a light one.
  const up = relativeLuminance(bg) < 0.5
  let best = hex

  for (let step = 1; step <= 40; step += 1) {
    const l = lch.l + (up ? step : -step) * 0.02

    if (l <= 0 || l >= 1) {
      break
    }

    best = oklchToHex({ ...lch, l })

    if (contrastRatio(best, bg) >= min) {
      return best
    }
  }

  return best
}
