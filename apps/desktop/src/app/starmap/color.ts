import { BLACK, MODE_DEFAULTS } from './constants'
import { clamp } from './geometry'
import type { Palette, Rgb } from './types'

// Theme tokens come through `color-mix()`/oklch, so getComputedStyle returns a
// non-rgb() string. Rasterize through a 1x1 canvas to get real sRGB bytes —
// naive string parsing of oklab()/color(srgb …) silently yields black.
let _probe: CanvasRenderingContext2D | null = null

export function resolveRgb(color: string): Rgb {
  if (!_probe) {
    const c = document.createElement('canvas')
    c.width = 1
    c.height = 1
    _probe = c.getContext('2d', { willReadFrequently: true })
  }

  if (!_probe) {
    return { b: 184, g: 163, r: 148 }
  }

  _probe.clearRect(0, 0, 1, 1)
  _probe.fillStyle = '#888888'
  _probe.fillStyle = color
  _probe.fillRect(0, 0, 1, 1)
  const d = _probe.getImageData(0, 0, 1, 1).data

  return { b: d[2], g: d[1], r: d[0] }
}

export function rgba(c: Rgb, a: number): string {
  return `rgba(${c.r},${c.g},${c.b},${a})`
}

export function mixRgb(a: Rgb, b: Rgb, t: number): Rgb {
  const p = clamp(t, 0, 1)

  return {
    b: Math.round(a.b + (b.b - a.b) * p),
    g: Math.round(a.g + (b.g - a.g) * p),
    r: Math.round(a.r + (b.r - a.r) * p)
  }
}

export function darken(c: Rgb, amount: number): Rgb {
  return mixRgb(c, BLACK, amount)
}

export function luminance(r: number, g: number, b: number): number {
  return (0.2126 * r + 0.7152 * g + 0.114 * b) / 255
}

function rgbToHsl(c: Rgb): [number, number, number] {
  const r = c.r / 255
  const g = c.g / 255
  const b = c.b / 255
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 2
  const d = max - min
  let h = 0
  let s = 0

  if (d) {
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
    h = (max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4) * 60
  }

  return [h, s, l]
}

function hslToRgb(h: number, s: number, l: number): Rgb {
  const hue = ((h % 360) + 360) % 360
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1))
  const m = l - c / 2

  const [r, g, b] =
    hue < 60
      ? [c, x, 0]
      : hue < 120
        ? [x, c, 0]
        : hue < 180
          ? [0, c, x]
          : hue < 240
            ? [0, x, c]
            : hue < 300
              ? [x, 0, c]
              : [c, 0, x]

  return { b: Math.round((b + m) * 255), g: Math.round((g + m) * 255), r: Math.round((r + m) * 255) }
}

// Complementary ink: rotate the source hue (the theme primary) and keep it vivid
// so memories read as a distinct color from skills, in any theme.
function complementaryInk(c: Rgb): Rgb {
  const [h, s, l] = rgbToHsl(c)

  return hslToRgb(h + 165, Math.max(s, 0.5), clamp(l, 0.5, 0.7))
}

// Memory ink: the complementary hue muted toward the overlay background so it
// reads as a distinct-but-quiet color (fake alpha), not a loud full-sat pop.
export function memoryInkFor(primary: Rgb, bg: Rgb): Rgb {
  return mixRgb(complementaryInk(primary), bg, 0.45)
}

// Resolve the theme-derived palette once per theme change — the resolveRgb probe
// does a getImageData readback, so this stays out of the per-frame path. Node
// groups borrow restrained tint from the theme; structure stays foreground ink.
export function computePalette(canvas: HTMLCanvasElement): Palette {
  const style = getComputedStyle(canvas)
  const fg = resolveRgb(style.color)
  const darkTheme = luminance(fg.r, fg.g, fg.b) > 0.55
  const base: Rgb = darkTheme ? { b: 255, g: 255, r: 255 } : { b: 0, g: 0, r: 0 }
  const primary = resolveRgb(style.getPropertyValue('--theme-primary').trim() || style.color)

  const bg = resolveRgb(
    style.getPropertyValue('--background').trim() ||
      style.getPropertyValue('--dt-background').trim() ||
      (darkTheme ? '#000' : '#fff')
  )

  return {
    // Band tint derives from the theme primary so rings read consistently in
    // both modes (foreground ink would go white on dark / black on light).
    bandInk: mixRgb(primary, base, darkTheme ? 0.3 : 0),
    base,
    bg,
    c: MODE_DEFAULTS[darkTheme ? 'dark' : 'light'],
    chipBg: darkTheme ? 'rgba(0,0,0,0.72)' : 'rgba(255,255,255,0.85)',
    darkTheme,
    inkInv: darkTheme ? 'rgba(0,0,0,1)' : 'rgba(255,255,255,1)',
    memoryInk: memoryInkFor(primary, bg),
    primary,
    skillInk: mixRgb(primary, base, darkTheme ? 0.12 : 0.18)
  }
}
