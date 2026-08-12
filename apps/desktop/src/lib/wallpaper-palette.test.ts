import { describe, expect, it } from 'vitest'

import { sanitizeWallpaperPalette } from './wallpaper-palette'

describe('wallpaper palette extraction', () => {
  it('sanitizes only complete six-digit cached palettes', () => {
    expect(sanitizeWallpaperPalette({ accent: '#AABBCC', dominant: '#102030' })).toEqual({
      accent: '#aabbcc',
      dominant: '#102030'
    })
    expect(sanitizeWallpaperPalette({ accent: 'red', dominant: '#102030' })).toBeNull()
    expect(sanitizeWallpaperPalette(null)).toBeNull()
  })
})
