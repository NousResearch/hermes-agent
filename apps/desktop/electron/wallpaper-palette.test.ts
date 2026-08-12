import assert from 'node:assert/strict'

import { test } from 'vitest'

import { extractWallpaperPaletteFromBgra } from './wallpaper-palette'

function pixels(entries: Array<{ alpha?: number; color: [number, number, number]; count: number }>): Uint8Array {
  return new Uint8Array(
    entries.flatMap(({ alpha = 255, color: [red, green, blue], count }) =>
      Array.from({ length: count }, () => [blue, green, red, alpha]).flat()
    )
  )
}

test('main-process palette extraction keeps a populous neutral and promotes a vivid accent', () => {
  const palette = extractWallpaperPaletteFromBgra(
    pixels([
      { color: [132, 136, 142], count: 80 },
      { color: [230, 54, 88], count: 20 }
    ])
  )

  assert.deepEqual(palette, { accent: '#e63658', dominant: '#84888e' })
})

test('main-process palette extraction ignores transparent pixels and handles monochrome images', () => {
  const palette = extractWallpaperPaletteFromBgra(
    pixels([
      { color: [72, 72, 72], count: 20 },
      { alpha: 0, color: [255, 0, 0], count: 200 }
    ])
  )

  assert.deepEqual(palette, { accent: '#484848', dominant: '#484848' })
  assert.equal(extractWallpaperPaletteFromBgra(pixels([{ alpha: 0, color: [1, 2, 3], count: 4 }])), null)
})
