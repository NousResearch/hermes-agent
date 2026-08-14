import { sanitizeWallpaperPalette, type WallpaperPalette } from './wallpaper-palette'

export const WALLPAPER_MODES = ['fill', 'fit', 'tile', 'center'] as const
export const WALLPAPER_MASK_SHAPES = ['ellipse', 'strip'] as const
export const WALLPAPER_PALETTE_MODES = ['auto', 'manual'] as const

export type WallpaperMode = (typeof WALLPAPER_MODES)[number]
export type WallpaperMaskShape = (typeof WALLPAPER_MASK_SHAPES)[number]
export type WallpaperPaletteMode = (typeof WALLPAPER_PALETTE_MODES)[number]

export const DEFAULT_MANUAL_WALLPAPER_PALETTE: WallpaperPalette = {
  accent: '#8174e8',
  dominant: '#7a7888'
}

export interface WallpaperPreferences {
  adaptiveTheme: boolean
  blur: number
  enabled: boolean
  manualPalette: WallpaperPalette | null
  mode: WallpaperMode
  opacity: number
  overlay: number
  overlayColor: string
  overlayHeight: number
  overlayShape: WallpaperMaskShape
  overlayWidth: number
  overlayX: number
  palette: WallpaperPalette | null
  paletteMode: WallpaperPaletteMode
  paletteSource: string
}

export const DEFAULT_WALLPAPER_PREFERENCES: WallpaperPreferences = {
  adaptiveTheme: false,
  blur: 0,
  enabled: true,
  manualPalette: null,
  mode: 'fill',
  opacity: 42,
  overlay: 48,
  overlayColor: '',
  overlayHeight: 145,
  overlayShape: 'ellipse',
  overlayWidth: 68,
  overlayX: 50,
  palette: null,
  paletteMode: 'auto',
  paletteSource: ''
}

export function wallpaperStorageKey(profile: string): string {
  return `hermes.desktop.wallpaper.v1.${encodeURIComponent(profile)}`
}

function boundedNumber(value: unknown, fallback: number, min: number, max: number): number {
  const number = Number(value)

  if (!Number.isFinite(number)) {
    return fallback
  }

  return Math.min(max, Math.max(min, Math.round(number)))
}

function sanitizedColor(value: unknown): string {
  if (value === '') {
    return ''
  }

  return typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value)
    ? value.toLowerCase()
    : DEFAULT_WALLPAPER_PREFERENCES.overlayColor
}

export function sanitizeWallpaperPreferences(value: unknown): WallpaperPreferences {
  const candidate = value && typeof value === 'object' && !Array.isArray(value) ? value : {}
  const record = candidate as Record<string, unknown>

  const mode = WALLPAPER_MODES.includes(record.mode as WallpaperMode)
    ? (record.mode as WallpaperMode)
    : DEFAULT_WALLPAPER_PREFERENCES.mode

  const overlayShape = WALLPAPER_MASK_SHAPES.includes(record.overlayShape as WallpaperMaskShape)
    ? (record.overlayShape as WallpaperMaskShape)
    : DEFAULT_WALLPAPER_PREFERENCES.overlayShape

  const paletteMode = WALLPAPER_PALETTE_MODES.includes(record.paletteMode as WallpaperPaletteMode)
    ? (record.paletteMode as WallpaperPaletteMode)
    : DEFAULT_WALLPAPER_PREFERENCES.paletteMode

  return {
    adaptiveTheme:
      typeof record.adaptiveTheme === 'boolean' ? record.adaptiveTheme : DEFAULT_WALLPAPER_PREFERENCES.adaptiveTheme,
    blur: boundedNumber(record.blur, DEFAULT_WALLPAPER_PREFERENCES.blur, 0, 24),
    enabled: typeof record.enabled === 'boolean' ? record.enabled : DEFAULT_WALLPAPER_PREFERENCES.enabled,
    manualPalette: sanitizeWallpaperPalette(record.manualPalette),
    mode,
    opacity: boundedNumber(record.opacity, DEFAULT_WALLPAPER_PREFERENCES.opacity, 0, 100),
    overlay: boundedNumber(record.overlay, DEFAULT_WALLPAPER_PREFERENCES.overlay, 0, 100),
    overlayColor: sanitizedColor(record.overlayColor),
    overlayHeight: boundedNumber(record.overlayHeight, DEFAULT_WALLPAPER_PREFERENCES.overlayHeight, 40, 200),
    overlayShape,
    overlayWidth: boundedNumber(record.overlayWidth, DEFAULT_WALLPAPER_PREFERENCES.overlayWidth, 30, 140),
    overlayX: boundedNumber(record.overlayX, DEFAULT_WALLPAPER_PREFERENCES.overlayX, 0, 100),
    palette: sanitizeWallpaperPalette(record.palette),
    paletteMode,
    paletteSource:
      typeof record.paletteSource === 'string' && record.paletteSource.length <= 512 ? record.paletteSource : ''
  }
}

export function wallpaperBackgroundProperties(mode: WallpaperMode): {
  position: string
  repeat: string
  size: string
} {
  switch (mode) {
    case 'fit':
      return { position: 'center', repeat: 'no-repeat', size: 'contain' }

    case 'tile':
      return { position: 'left top', repeat: 'repeat', size: 'auto' }

    case 'center':
      return { position: 'center', repeat: 'no-repeat', size: 'auto' }

    default:
      return { position: 'center', repeat: 'no-repeat', size: 'cover' }
  }
}

export function wallpaperMaskImage(
  overlayShape: WallpaperMaskShape,
  overlayX: number,
  overlayWidth: number,
  overlayHeight: number
): string {
  const position = boundedNumber(overlayX, DEFAULT_WALLPAPER_PREFERENCES.overlayX, 0, 100)
  const width = boundedNumber(overlayWidth, DEFAULT_WALLPAPER_PREFERENCES.overlayWidth, 30, 140)
  const height = boundedNumber(overlayHeight, DEFAULT_WALLPAPER_PREFERENCES.overlayHeight, 40, 200)

  if (overlayShape === 'strip') {
    const outerHalf = width / 2
    const innerHalf = width * 0.32

    return `linear-gradient(90deg, transparent ${position - outerHalf}%, #000 ${position - innerHalf}%, #000 ${position + innerHalf}%, transparent ${position + outerHalf}%)`
  }

  return `radial-gradient(ellipse ${width}% ${height}% at ${position}% 50%, #000 0%, #000 28%, transparent 78%)`
}
