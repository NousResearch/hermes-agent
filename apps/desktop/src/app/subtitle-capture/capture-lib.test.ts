import { describe, expect, it } from 'vitest'

import {
  BRIGHT_LUMA,
  brightMaskHash,
  cropRectFor,
  hammingDistance,
  HASH_HEIGHT,
  HASH_WIDTH,
  SAME_TEXT_MAX_DISTANCE,
  SHIP_MAX_WIDTH,
  shipSize
} from './capture-lib'

/** RGBA frame builder: dark background at `backgroundLuma`, plus white "text"
 *  rectangles. Deliberately synthetic — the contract under test is "the hash
 *  follows the bright mask, not the background". */
function frame(
  width: number,
  height: number,
  backgroundLuma: number,
  textRects: Array<{ h: number; w: number; x: number; y: number }>
): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4)

  for (let index = 0; index < width * height; index += 1) {
    rgba[index * 4] = backgroundLuma
    rgba[index * 4 + 1] = backgroundLuma
    rgba[index * 4 + 2] = backgroundLuma
    rgba[index * 4 + 3] = 255
  }

  for (const rect of textRects) {
    for (let y = rect.y; y < rect.y + rect.h; y += 1) {
      for (let x = rect.x; x < rect.x + rect.w; x += 1) {
        const offset = (y * width + x) * 4

        rgba[offset] = 255
        rgba[offset + 1] = 255
        rgba[offset + 2] = 255
      }
    }
  }

  return rgba
}

const W = 320
const H = 80
const LINE = [{ h: 20, w: 200, x: 60, y: 30 }]

describe('brightMaskHash', () => {
  it('packs one bit per cell', () => {
    expect(brightMaskHash(frame(W, H, 20, []), W, H)).toHaveLength((HASH_WIDTH * HASH_HEIGHT) / 4)
  })

  it('ignores background motion: same text over different dark scenes hashes the same', () => {
    const darkScene = brightMaskHash(frame(W, H, 15, LINE), W, H)
    const lighterScene = brightMaskHash(frame(W, H, 90, LINE), W, H)

    expect(hammingDistance(darkScene, lighterScene)).toBeLessThanOrEqual(SAME_TEXT_MAX_DISTANCE)
  })

  it('fires when the text changes or disappears', () => {
    const line = brightMaskHash(frame(W, H, 15, LINE), W, H)
    const otherLine = brightMaskHash(frame(W, H, 15, [{ h: 20, w: 120, x: 20, y: 55 }]), W, H)
    const empty = brightMaskHash(frame(W, H, 15, []), W, H)

    expect(hammingDistance(line, otherLine)).toBeGreaterThan(SAME_TEXT_MAX_DISTANCE)
    expect(hammingDistance(line, empty)).toBeGreaterThan(SAME_TEXT_MAX_DISTANCE)
  })

  it('the luma threshold separates subtitle-white from mid-gray background', () => {
    // A background just under the threshold must not read as text.
    const almostBright = brightMaskHash(frame(W, H, BRIGHT_LUMA - 5, []), W, H)
    const empty = brightMaskHash(frame(W, H, 15, []), W, H)

    expect(hammingDistance(almostBright, empty)).toBe(0)
  })
})

describe('hammingDistance', () => {
  it('counts flipped bits across hex nibbles', () => {
    expect(hammingDistance('00', '00')).toBe(0)
    expect(hammingDistance('0f', '00')).toBe(4)
    expect(hammingDistance('ff', '00')).toBe(8)
  })

  it('treats a length mismatch as maximally distant', () => {
    expect(hammingDistance('00', '000')).toBeGreaterThan(SAME_TEXT_MAX_DISTANCE)
  })
})

describe('cropRectFor', () => {
  const FRACTIONS = { height: 0.25, left: 0.1, top: 0.7, width: 0.8 }

  it('scales fractions by the live video size', () => {
    expect(cropRectFor(FRACTIONS, 2000, 1000)).toEqual({ height: 250, width: 1600, x: 200, y: 700 })
  })

  it('clamps to the frame and rejects degenerate results', () => {
    expect(cropRectFor({ height: 0.5, left: 0.9, top: 0.9, width: 0.5 }, 1000, 500)).toEqual({
      height: 50,
      width: 100,
      x: 900,
      y: 450
    })
    expect(cropRectFor(FRACTIONS, 0, 0)).toBeNull()
    expect(cropRectFor({ height: 0.001, left: 0, top: 0.99, width: 1 }, 1000, 500)).toBeNull()
  })
})

describe('shipSize', () => {
  it('passes small crops through and downscales wide ones preserving aspect', () => {
    expect(shipSize({ height: 100, width: 800, x: 0, y: 0 })).toEqual({ height: 100, width: 800 })

    const shipped = shipSize({ height: 300, width: 2560, x: 0, y: 0 })

    expect(shipped.width).toBe(SHIP_MAX_WIDTH)
    expect(shipped.height).toBe(150)
  })
})
