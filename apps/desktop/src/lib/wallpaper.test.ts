import { beforeEach, describe, expect, it } from 'vitest'

import { readJson, writeJson } from './storage'
import {
  DEFAULT_WALLPAPER_PREFERENCES,
  sanitizeWallpaperPreferences,
  wallpaperBackgroundProperties,
  wallpaperMaskImage,
  wallpaperStorageKey
} from './wallpaper'

describe('wallpaper preferences', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('validates persisted values and clamps hot controls', () => {
    expect(
      sanitizeWallpaperPreferences({
        adaptiveTheme: 'yes',
        blur: 200,
        enabled: false,
        manualPalette: { accent: '#123456', dominant: 'gray' },
        mode: 'unknown',
        opacity: -4,
        overlay: 72.4,
        overlayColor: 'red',
        overlayFeather: 999,
        overlayHeight: 999,
        overlayShape: 'rectangle',
        overlayWidth: 0,
        overlayX: Number.NaN,
        palette: { accent: 'blue', dominant: '#123456' },
        paletteMode: 'custom',
        paletteSource: 99
      })
    ).toEqual({
      adaptiveTheme: false,
      blur: 24,
      enabled: false,
      manualPalette: null,
      mode: 'fill',
      opacity: 0,
      overlay: 72,
      overlayColor: '',
      overlayFeather: 100,
      overlayHeight: 200,
      overlayShape: 'ellipse',
      overlayWidth: 30,
      overlayX: 50,
      palette: null,
      paletteMode: 'auto',
      paletteSource: ''
    })
    expect(
      sanitizeWallpaperPreferences({
        adaptiveTheme: true,
        manualPalette: { accent: '#DDEEFF', dominant: '#445566' },
        overlayColor: '#A1B2C3',
        overlayFeather: 12,
        overlayShape: 'strip',
        palette: { accent: '#AABBCC', dominant: '#102030' },
        paletteMode: 'manual',
        paletteSource: 'hermes-wallpaper://asset/example?v=1'
      })
    ).toMatchObject({
      adaptiveTheme: true,
      manualPalette: { accent: '#ddeeff', dominant: '#445566' },
      overlayColor: '#a1b2c3',
      overlayFeather: 12,
      overlayShape: 'strip',
      palette: { accent: '#aabbcc', dominant: '#102030' },
      paletteMode: 'manual',
      paletteSource: 'hermes-wallpaper://asset/example?v=1'
    })
    expect(sanitizeWallpaperPreferences(null)).toEqual(DEFAULT_WALLPAPER_PREFERENCES)
  })

  it('keeps presentation preferences separate for each profile', () => {
    const defaultKey = wallpaperStorageKey('default')
    const workKey = wallpaperStorageKey('work')

    writeJson(defaultKey, { ...DEFAULT_WALLPAPER_PREFERENCES, opacity: 20 })
    writeJson(workKey, { ...DEFAULT_WALLPAPER_PREFERENCES, opacity: 80 })

    expect(readJson<{ opacity: number }>(defaultKey)?.opacity).toBe(20)
    expect(readJson<{ opacity: number }>(workKey)?.opacity).toBe(80)
  })

  it('maps every rendering mode and moves the readability mask', () => {
    expect(wallpaperBackgroundProperties('fill')).toEqual({ position: 'center', repeat: 'no-repeat', size: 'cover' })
    expect(wallpaperBackgroundProperties('fit').size).toBe('contain')
    expect(wallpaperBackgroundProperties('tile').repeat).toBe('repeat')
    expect(wallpaperBackgroundProperties('center').size).toBe('auto')
    const ellipse = wallpaperMaskImage('ellipse', 10, 80, 160)

    expect(ellipse).toContain('ellipse 80% 160% at 10% 50%')
    expect(ellipse).toContain('#000 23%')
    expect(ellipse).toContain('transparent 83%')
    expect(wallpaperMaskImage('ellipse', 90, 80, 160)).toContain('at 90% 50%')
    expect(wallpaperMaskImage('ellipse', 500, 500, 500)).toContain('ellipse 140% 200% at 100% 50%')

    const strip = wallpaperMaskImage('strip', 50, 40, 160)

    expect(strip).toContain('linear-gradient(90deg')
    expect(strip).toContain('transparent 30%')
    expect(strip).toContain('transparent 70%')
  })

  it('smooths mask edges while preserving a sharp zero-feather option', () => {
    const sharpEllipse = wallpaperMaskImage('ellipse', 50, 80, 160, 0)
    const softEllipse = wallpaperMaskImage('ellipse', 50, 80, 160, 100)
    const sharpStrip = wallpaperMaskImage('strip', 50, 40, 160, 0)
    const softStrip = wallpaperMaskImage('strip', 50, 40, 160, 100)

    expect(sharpEllipse).toContain('#000 53%, transparent 53%')
    expect(softEllipse).toContain('#000 3%')
    expect(softEllipse.match(/rgba\(/g)).toHaveLength(11)
    expect(softEllipse).toContain('transparent 100%')
    expect(sharpStrip).toContain('transparent 30%, #000 30%')
    expect(softStrip.match(/rgba\(/g)).toHaveLength(22)
  })
})
