/**
 * Theme Forge — color math, palette extraction, and theme synthesis.
 *
 * Everything here is renderer-safe pure logic except `extractPalette` /
 * `thumbOf`, which need a canvas. The verbatim contract is the whole point:
 * slot 1 IS the background, slot 2 IS the text color (UI + terminal), and the
 * first chromatic swatch IS the accent — the exact color the user places is
 * the exact color that lands in the theme. Secondary tokens (muted text,
 * borders) still get contrast-safe derivation.
 */

// ── types ───────────────────────────────────────────────────────────────────
// Structural matches of the app's `DesktopTheme` / `DesktopThemeColors` /
// `DesktopTerminalPalette` (src/themes/types.ts) — the plugin fence keeps
// bundled plugins on `@hermes/plugin-sdk` imports only, so the shapes are
// redeclared here and passed through `THEMES_AREA` as contribution data.

export interface ForgeColors {
  background: string
  foreground: string
  card: string
  cardForeground: string
  muted: string
  mutedForeground: string
  popover: string
  popoverForeground: string
  primary: string
  primaryForeground: string
  secondary: string
  secondaryForeground: string
  accent: string
  accentForeground: string
  border: string
  input: string
  ring: string
  midground?: string
  composerRing?: string
  destructive: string
  destructiveForeground: string
  sidebarBackground?: string
  sidebarBorder?: string
  userBubble?: string
  userBubbleBorder?: string
}

export interface ForgeTerminalPalette {
  foreground?: string
  cursor?: string
  selectionBackground?: string
  black?: string
  red?: string
  green?: string
  yellow?: string
  blue?: string
  magenta?: string
  cyan?: string
  white?: string
  brightBlack?: string
  brightRed?: string
  brightGreen?: string
  brightYellow?: string
  brightBlue?: string
  brightMagenta?: string
  brightCyan?: string
  brightWhite?: string
}

export interface ForgeTheme {
  name: string
  label: string
  description: string
  colors: ForgeColors
  darkColors: ForgeColors
  terminal: ForgeTerminalPalette
  darkTerminal: ForgeTerminalPalette
}

export interface Swatch {
  hex: string
  hsl: Hsl
  weight: number
}

export interface Hsl {
  h: number
  s: number
  l: number
}

export type ForgeMode = 'dark' | 'light'

/** A persisted forged theme — the `ctx.storage` entry shape. */
export interface ForgeEntry {
  name: string
  label: string
  mode: ForgeMode
  swatches: Swatch[]
  theme: ForgeTheme
  /** Downscaled JPEG data-URL of the source image (reforge after restarts). */
  source: string | null
  forgedAt: number
}

// ── color math ──────────────────────────────────────────────────────────────

export const rgbToHex = (r: number, g: number, b: number): string =>
  '#' +
  [r, g, b]
    .map(n =>
      Math.round(Math.min(255, Math.max(0, n)))
        .toString(16)
        .padStart(2, '0')
    )
    .join('')

export const hexToRgb = (hex: string): [number, number, number] | null => {
  const c = String(hex).trim().replace(/^#/, '')

  if (!/^[0-9a-f]{6}$/i.test(c)) {
    return null
  }

  return [0, 2, 4].map(i => parseInt(c.slice(i, i + 2), 16)) as [number, number, number]
}

export const mix = (a: string, b: string, t: number): string => {
  const A = hexToRgb(a)
  const B = hexToRgb(b)

  return A && B ? rgbToHex(A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, A[2] + (B[2] - A[2]) * t) : a
}

const lin = (c: number): number => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)

export const luminance = (hex: string): number => {
  const rgb = hexToRgb(hex)

  if (!rgb) {
    return 0
  }

  const [r, g, b] = rgb.map(v => lin(v / 255))

  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

export const contrast = (a: string, b: string): number => {
  const la = luminance(a)
  const lb = luminance(b)

  return la >= lb ? (la + 0.05) / (lb + 0.05) : (lb + 0.05) / (la + 0.05)
}

export const readableOn = (bg: string): string => (luminance(bg) > 0.58 ? '#161616' : '#ffffff')

/**
 * Tripwire floor for the verbatim foreground (slot-2) color. WCAG body-text
 * target is 4.5:1, but the user's whole point of a verbatim slot-2 is to place
 * an exact accent/text color — so the floor is deliberately the LARGE-TEXT bar
 * (3:1), a "you can't read this at all" tripwire, not a readability rule.
 * ensureContrast is a no-op above it, so readable low-contrast accents pass
 * through untouched; only a near-black-on-near-black hard illegibility nudges
 * toward the correct polarity just enough to clear the bar. Keeps creative
 * freedom while guaranteeing the theme is never literally unreadable.
 */
export const FORGE_TEXT_FLOOR = 3

export const ensureContrast = (color: string, bg: string, min: number): string => {
  if (contrast(color, bg) >= min) {
    return color
  }

  const toward = luminance(bg) < 0.5 ? '#ffffff' : '#000000'
  let best = color

  for (let t = 0.1; t <= 1.001; t += 0.1) {
    const c = mix(color, toward, t)

    if (contrast(c, bg) > contrast(best, bg)) {
      best = c
    }

    if (contrast(c, bg) >= min) {
      return c
    }
  }

  return readableOn(bg)
}

export function rgbToHsl(r: number, g: number, b: number): Hsl {
  r /= 255
  g /= 255
  b /= 255
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 2
  let h = 0
  let s = 0

  if (max !== min) {
    const d = max - min
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min)

    switch (max) {
      case r:
        h = ((g - b) / d + (g < b ? 6 : 0)) / 6

        break

      case g:
        h = ((b - r) / d + 2) / 6

        break

      default:
        h = ((r - g) / d + 4) / 6
    }
  }

  return { h, s, l }
}

export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  let r: number
  let g: number
  let b: number

  if (s === 0) {
    r = g = b = l
  } else {
    const hue2rgb = (p: number, q: number, t: number): number => {
      if (t < 0) {
        t += 1
      }

      if (t > 1) {
        t -= 1
      }

      if (t < 1 / 6) {
        return p + (q - p) * 6 * t
      }

      if (t < 1 / 2) {
        return q
      }

      if (t < 2 / 3) {
        return p + (q - p) * (2 / 3 - t) * 6
      }

      return p
    }

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s
    const p = 2 * l - q
    r = hue2rgb(p, q, h + 1 / 3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1 / 3)
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
}

export const hexToHsl = (hex: string): Hsl | null => {
  const rgb = hexToRgb(hex)

  return rgb ? rgbToHsl(rgb[0], rgb[1], rgb[2]) : null
}

export const hslToHex = (h: number, s: number, l: number): string => {
  const [r, g, b] = hslToRgb(h, s, l)

  return rgbToHex(r, g, b)
}

// ── palette extraction: median-cut ──────────────────────────────────────────

export function extractPalette(imgEl: HTMLImageElement, maxColors = 12): Swatch[] {
  const w = imgEl.naturalWidth || imgEl.width
  const h = imgEl.naturalHeight || imgEl.height
  const side = 256
  const scale = Math.min(1, side / Math.max(w, h))
  const cw = Math.max(1, Math.round(w * scale))
  const ch = Math.max(1, Math.round(h * scale))

  const canvas = document.createElement('canvas')
  canvas.width = cw
  canvas.height = ch
  const g2d = canvas.getContext('2d', { willReadFrequently: true })

  if (!g2d) {
    throw new Error('Canvas 2D context unavailable')
  }

  g2d.drawImage(imgEl, 0, 0, cw, ch)

  const data = g2d.getImageData(0, 0, cw, ch).data
  const px: number[][] = []

  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 128) {
      continue
    }

    px.push([data[i], data[i + 1], data[i + 2]])
  }

  if (px.length < 8) {
    throw new Error('Image has no usable pixels (too small or transparent)')
  }

  let boxes = [px]

  while (boxes.length < maxColors) {
    let bi = 0
    let bestRange = -1
    let bestCh = 0
    boxes.forEach((box, idx) => {
      for (let ch2 = 0; ch2 < 3; ch2++) {
        let lo = 255
        let hi = 0

        for (const p of box) {
          if (p[ch2] < lo) {
            lo = p[ch2]
          }

          if (p[ch2] > hi) {
            hi = p[ch2]
          }
        }

        if (hi - lo > bestRange) {
          bestRange = hi - lo
          bestCh = ch2
          bi = idx
        }
      }
    })

    if (bestRange <= 0) {
      break
    }

    const box = boxes[bi]
    box.sort((a, b) => a[bestCh] - b[bestCh])
    const mid = Math.floor(box.length / 2)
    boxes.splice(bi, 1, box.slice(0, mid), box.slice(mid))
  }

  return boxes
    .filter(b => b.length > 0)
    .map(b => {
      let r = 0
      let g = 0
      let bl = 0

      for (const p of b) {
        r += p[0]
        g += p[1]
        bl += p[2]
      }

      const n = b.length
      const hex = rgbToHex(r / n, g / n, bl / n)

      return { hex, hsl: rgbToHsl(r / n, g / n, bl / n), weight: n }
    })
    .sort((a, b) => b.weight - a.weight)
}

/** Downscale to a small JPEG data-URL so sources survive restarts cheaply. */
export function thumbOf(imgEl: HTMLImageElement): string {
  const w = imgEl.naturalWidth || imgEl.width
  const h = imgEl.naturalHeight || imgEl.height
  const side = 128
  const s = Math.min(1, side / Math.max(w, h))
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(w * s))
  canvas.height = Math.max(1, Math.round(h * s))
  const g = canvas.getContext('2d')

  if (!g) {
    throw new Error('Canvas 2D context unavailable')
  }

  g.drawImage(imgEl, 0, 0, canvas.width, canvas.height)

  return canvas.toDataURL('image/jpeg', 0.75)
}

// ── theme synthesis ─────────────────────────────────────────────────────────
// Swatch ORDER is user-editable: index 0 seeds the background, remaining
// swatches rank as accent priority (most-chromatic slot first).

export const slugify = (s: string): string =>
  String(s)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40) || 'forged'

export function ansiPalette(ordered: Swatch[], bg: string, fgSeed: string | null): ForgeTerminalPalette {
  const bgL = luminance(bg)
  const darkBg = bgL < 0.5
  // accent priority = swatches after the bg seed, chromatic ones first
  const chroma = ordered.slice(1).filter(c => c.hsl.s > 0.12)
  const fallbacks = ordered.slice(1).filter(c => c.hsl.s <= 0.12)
  const pool = [...chroma, ...fallbacks]

  const pick = (hLo: number, hHi: number, fi: number): string => {
    const hit = chroma.find(c => {
      const h = c.hsl.h * 360

      return h >= hLo && h < hHi
    })

    const alt = pool.find((c, i) => i === fi % Math.max(1, pool.length))

    return (hit || alt || ordered[0]).hex
  }

  const tune = (hex: string, lift: number): string =>
    darkBg ? ensureContrast(mix(hex, '#ffffff', lift), bg, 3) : ensureContrast(mix(hex, '#000000', lift * 0.8), bg, 3)

  const black = darkBg ? mix(bg, '#ffffff', 0.08) : mix(bg, '#000000', 0.55)
  const white = darkBg ? mix(bg, '#ffffff', 0.85) : mix(bg, '#000000', 0.08)

  return {
    // Terminal body text follows the slot-2 swatch VERBATIM — the exact color
    // the user places is the terminal foreground. Tripwire floor applies: a
    // no-op above FORGE_TEXT_FLOOR, nudges only when the pair is unreadable.
    foreground: fgSeed ? ensureContrast(fgSeed, bg, FORGE_TEXT_FLOOR) : readableOn(bg),
    cursor: tune(pick(150, 260, 0), 0.2),
    selectionBackground: darkBg ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.14)',
    black,
    red: tune(pick(345, 381, 0), 0.1),
    green: tune(pick(80, 160, 1), 0.1),
    yellow: tune(pick(40, 70, 2), 0.1),
    blue: tune(pick(200, 260, 3), 0.1),
    magenta: tune(pick(280, 345, 4), 0.1),
    cyan: tune(pick(160, 200, 5), 0.1),
    white,
    brightBlack: mix(black, white, darkBg ? 0.35 : 0.25),
    brightRed: mix(tune(pick(345, 381, 0), 0.1), white, darkBg ? 0.25 : 0),
    brightGreen: mix(tune(pick(80, 160, 1), 0.1), white, darkBg ? 0.25 : 0),
    brightYellow: mix(tune(pick(40, 70, 2), 0.1), white, darkBg ? 0.25 : 0),
    brightBlue: mix(tune(pick(200, 260, 3), 0.1), white, darkBg ? 0.25 : 0),
    brightMagenta: mix(tune(pick(280, 345, 4), 0.1), white, darkBg ? 0.25 : 0),
    brightCyan: mix(tune(pick(160, 200, 5), 0.1), white, darkBg ? 0.25 : 0),
    brightWhite: mix(white, darkBg ? '#ffffff' : '#000000', darkBg ? 0.4 : 0.18)
  }
}

export function buildColorsFromPalette(ordered: Swatch[], wantDark: boolean): ForgeColors {
  const seed = ordered[0] || {
    hex: wantDark ? '#101014' : '#fafafa',
    hsl: { h: 0, s: 0, l: wantDark ? 0.06 : 0.97 },
    weight: 0
  }

  const rest = ordered.slice(1)
  const byLum = [...rest].sort((a, b) => a.hsl.l - b.hsl.l)

  const chromaRank = [...rest].sort(
    (a, b) => b.hsl.s * (1 - Math.abs(b.hsl.l - 0.5)) - a.hsl.s * (1 - Math.abs(a.hsl.l - 0.5))
  )

  // accent = first chromatic swatch in USER order (priority), else chroma rank
  const userAccent = rest.find(c => c.hsl.s > 0.12)
  const accentRaw = userAccent || chromaRank[0] || seed

  // ── Background: slot 1 IS the background color, VERBATIM. No lightness
  // enforcement, no mix-toward-black/white: the color the user places in
  // slot 1 is exactly the background the theme uses.
  const background = seed.hex

  // Foreground: slot 2 IS the foreground/text color, starting point. The
  // exact color the user places in slot 2 is the text color the theme uses
  // (UI + terminal) — this is the fix for "swapped swatches and text never
  // visibly changed": the old code blended every slot-2 seed toward
  // near-white/near-black for contrast, so the swap never showed THAT color.
  //
  // Tripwire floor (below): the verbatim contract is preserved UNLESS the pair
  // is genuinely unreadable (near-black text on a near-black bg). ensureContrast
  // is a no-op above the floor, so deliberate, readable low-contrast choices
  // pass through untouched — freedom kept. Only a hard illegibility (contrast
  // < FORGE_TEXT_FLOOR) nudges toward the correct polarity, just enough to clear
  // the floor. This is the backstop the user asked for after landing on a
  // super-dark, unreadable theme.
  let foreground: string

  if (ordered.length >= 2) {
    foreground = ordered[1].hex
  } else {
    if (wantDark) {
      foreground = (byLum[byLum.length - 1] || seed).hex

      if (luminance(foreground) < 0.55) {
        foreground = mix(foreground, '#ffffff', 0.75)
      }
    } else {
      foreground = (byLum[0] || seed).hex

      if (luminance(foreground) > 0.35) {
        foreground = mix(foreground, '#060608', 0.7)
      }
    }
  }

  foreground = ensureContrast(foreground, background, FORGE_TEXT_FLOOR)

  const accentSafe = accentRaw.hex
  const card = wantDark ? mix(background, '#ffffff', 0.045) : mix(background, '#000000', 0.015)
  const muted = wantDark ? mix(background, '#ffffff', 0.07) : mix(background, '#000000', 0.045)
  const mutedFg = ensureContrast(mix(foreground, background, 0.42), background, 4.5)
  const border = wantDark ? mix(background, '#ffffff', 0.14) : mix(background, '#000000', 0.13)

  return {
    background,
    // Slot-2 color is the text color, verbatim — no contrast re-mix so the
    // swap shows THAT color on screen. (Readability is the user's call now.)
    foreground,
    card,
    cardForeground: foreground,
    muted,
    mutedForeground: mutedFg,
    popover: card,
    popoverForeground: foreground,
    primary: accentSafe,
    primaryForeground: readableOn(accentSafe),
    secondary: mix(muted, accentSafe, 0.12),
    secondaryForeground: ensureContrast(foreground, muted, 5),
    accent: mix(muted, accentSafe, 0.22),
    accentForeground: ensureContrast(foreground, mix(muted, accentSafe, 0.22), 5),
    border,
    input: border,
    ring: accentSafe,
    midground: accentSafe,
    composerRing: accentSafe,
    destructive: ensureContrast(wantDark ? '#c0473a' : '#c72e4d', background, 3),
    destructiveForeground: '#ffffff',
    sidebarBackground: wantDark ? mix(background, '#000000', 0.16) : mix(background, '#000000', 0.03),
    sidebarBorder: wantDark ? mix(border, '#ffffff', 0.02) : border,
    userBubble: mix(muted, accentSafe, 0.16),
    userBubbleBorder: mix(border, accentSafe, 0.3)
  }
}

/** Build the full DesktopTheme from an ORDERED swatch list. */
export function synthesize(ordered: Swatch[], meta: { name: string; label: string; mode: ForgeMode }): ForgeTheme {
  const darkColors = buildColorsFromPalette(ordered, true)
  const lightColors = buildColorsFromPalette(ordered, false)
  const primary = meta.mode === 'light' ? lightColors : darkColors
  // Slot 2 (index 1) is the TEXT seed — feed its raw hue to the terminal
  // palette so the terminal's body text follows the same swatch as the UI.
  const textSeed = ordered.length >= 2 ? ordered[1].hex : null

  return {
    name: meta.name,
    label: meta.label,
    description: 'Forged from an image · theme-forge plugin',
    colors: primary,
    darkColors,
    terminal: ansiPalette(ordered, primary.background, textSeed),
    darkTerminal: ansiPalette(ordered, darkColors.background, textSeed)
  }
}

/**
 * Recover a swatch list from a theme's own tokens — for v1-era entries that
 * never stored their extracted palette. Order follows the tray's semantics:
 * slot 1 = background seed, rest = accent priority.
 */
export function deriveSwatches(theme: ForgeTheme): Swatch[] {
  const c = theme.darkColors || theme.colors || ({} as ForgeColors)
  const t = theme.darkTerminal || theme.terminal || ({} as ForgeTerminalPalette)

  const candidates = [
    c.background,
    c.primary,
    c.foreground,
    t.red,
    t.green,
    t.blue,
    t.yellow,
    t.magenta,
    t.cyan,
    c.destructive,
    c.accent,
    c.secondary
  ]

  const seen = new Set<string>()
  const out: Swatch[] = []

  for (const hex of candidates) {
    if (typeof hex !== 'string' || !/^#[0-9a-f]{6}$/i.test(hex)) {
      continue
    }

    const key = hex.toLowerCase()

    if (seen.has(key)) {
      continue
    }

    seen.add(key)
    const rgb = hexToRgb(hex)

    if (!rgb) {
      continue
    }

    out.push({ hex, hsl: rgbToHsl(rgb[0], rgb[1], rgb[2]), weight: 1000 - out.length })

    if (out.length >= 8) {
      break
    }
  }

  return out
}

/**
 * The plugin used to auto-prepend 'Forge · ' to every theme label. Sleek
 * mode: strip ONLY that exact auto-injected prefix. Names that carry 'Forge'
 * as part of the actual name ('Dark Forge', 'Forge Midnight') are untouched.
 * Falls back to the original if stripping would empty the label.
 */
export const stripForgePrefix = (label: string | null | undefined): string => {
  const raw = String(label || '')

  return raw.replace(/^\s*forge\s*[·•]\s*/i, '').trim() || raw
}

/** Strict hex parse: 3- or 6-digit (#abc / #aabbcc) → normalized 6-digit, else null. */
export const parseHexStrict = (v: string): string | null => {
  const c = String(v).trim().replace(/^#/, '')

  if (/^[0-9a-f]{3}$/i.test(c)) {
    return (
      '#' +
      c
        .split('')
        .map(x => x + x)
        .join('')
    )
  }

  if (/^[0-9a-f]{6}$/i.test(c)) {
    return '#' + c.toLowerCase()
  }

  return null
}

// ── forge pipeline (canvas-dependent) ───────────────────────────────────────

export async function loadImageFromUrl(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const el = new Image()
    el.onload = () => resolve(el)
    el.onerror = () => reject(new Error('Could not decode that image'))
    el.src = url
  })
}

export async function forgeTheme(file: File, mode: ForgeMode): Promise<ForgeEntry> {
  const url = URL.createObjectURL(file)

  try {
    const img = await loadImageFromUrl(url)
    const palette = extractPalette(img, 12)
    const baseName = file.name.replace(/\.[a-z0-9]+$/i, '')
    const slug = slugify(baseName)
    const label = baseName.length > 24 ? baseName.slice(0, 24) + '…' : baseName
    const ordered = [...palette].sort((a, b) => b.weight - a.weight).slice(0, 8)
    const themeName = `forge-${slug}`

    return {
      name: themeName,
      label,
      mode,
      swatches: ordered,
      theme: synthesize(ordered, { name: themeName, label, mode }),
      source: thumbOf(img),
      forgedAt: Date.now()
    }
  } finally {
    URL.revokeObjectURL(url)
  }
}
