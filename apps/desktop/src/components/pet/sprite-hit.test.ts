import { describe, expect, it } from 'vitest'

import { ALPHA_HIT_THRESHOLD, type SpriteHitSurface, spriteSolidAt } from './sprite-hit'

function surface(alphaAt: (x: number, y: number) => number, opts?: Partial<SpriteHitSurface>): SpriteHitSurface {
  return {
    width: 4,
    height: 2,
    getBoundingClientRect: () => ({ left: 10, top: 20, width: 40, height: 20 }),
    getContext: () => ({
      getImageData: (x, y) => ({ data: [0, 0, 0, alphaAt(x, y)] })
    }),
    ...opts
  }
}

describe('spriteSolidAt', () => {
  it('treats alpha at the threshold as solid and just below as a miss', () => {
    const solid = surface(() => ALPHA_HIT_THRESHOLD)
    const faint = surface(() => ALPHA_HIT_THRESHOLD - 1)

    expect(spriteSolidAt(solid, 10, 20)).toBe(true)
    expect(spriteSolidAt(faint, 10, 20)).toBe(false)
  })

  it('maps the viewport point onto the bitmap, so padding in the cell is a miss', () => {
    // 40 css px / 4 bitmap px → 10 css px per cell. Pixel (2, 0) is the only solid one.
    const sheet = surface((x, y) => (x === 2 && y === 0 ? 255 : 0))

    expect(spriteSolidAt(sheet, 30, 20)).toBe(true) // css x 30 → bitmap x 2
    expect(spriteSolidAt(sheet, 15, 20)).toBe(false) // css x 15 → bitmap x 0, transparent
  })

  it('does not claim a point outside the canvas box', () => {
    expect(
      spriteSolidAt(
        surface(() => 255),
        9,
        20
      )
    ).toBe(false)
    expect(
      spriteSolidAt(
        surface(() => 255),
        50,
        20
      )
    ).toBe(false)
  })

  it('returns null when the pixel cannot be read, so callers can fail open or closed', () => {
    expect(
      spriteSolidAt(
        surface(() => 255, { width: 0 }),
        10,
        20
      )
    ).toBeNull()
    expect(
      spriteSolidAt(
        surface(() => 255, { getContext: () => null }),
        10,
        20
      )
    ).toBeNull()
    expect(
      spriteSolidAt(
        surface(() => 255, {
          getContext: () => ({
            getImageData: () => {
              throw new Error('tainted')
            }
          })
        }),
        10,
        20
      )
    ).toBeNull()
  })
})
