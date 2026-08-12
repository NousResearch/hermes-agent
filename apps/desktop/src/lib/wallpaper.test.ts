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
        overlayShape: 'strip',
        palette: { accent: '#AABBCC', dominant: '#102030' },
        paletteMode: 'manual',
        paletteSource: 'hermes-wallpaper://asset/example?v=1'
      })
    ).toMatchObject({
      adaptiveTheme: true,
      manualPalette: { accent: '#ddeeff', dominant: '#445566' },
      overlayColor: '#a1b2c3',
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
    expect(wallpaperMaskImage('ellipse', 10, 80, 160)).toContain('ellipse 80% 160% at 10% 50%')
    expect(wallpaperMaskImage('ellipse', 90, 80, 160)).toContain('at 90% 50%')
    expect(wallpaperMaskImage('ellipse', 500, 500, 500)).toContain('ellipse 140% 200% at 100% 50%')

    const strip = wallpaperMaskImage('strip', 50, 40, 160)

    expect(strip).toContain('linear-gradient(90deg')
    expect(strip).toContain('transparent 30%')
    expect(strip).toContain('transparent 70%')
  })
})
