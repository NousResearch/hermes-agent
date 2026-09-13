import { describe, expect, it } from 'vitest'

import { type ContrastBitmap, detectHorizontalContrastSurface } from './pet-overlay-contrast'

const bitmap = (width: number, height: number, pixel: (x: number, y: number) => number): ContrastBitmap => {
  const data = new Uint8Array(width * height * 4)

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = pixel(x, y)
      const offset = (y * width + x) * 4

      data[offset] = value
      data[offset + 1] = value
      data[offset + 2] = value
      data[offset + 3] = 255
    }
  }

  return { data, height, width }
}

describe('pet overlay visual contrast surfaces', () => {
  it('detects a wide horizontal edge below the pet', () => {
    const frame = bitmap(320, 180, (x, y) => (x >= 40 && x < 280 && y >= 92 ? 210 : 45))

    const surface = detectHorizontalContrastSurface(frame, {
      centerX: 160,
      footY: 58,
      fromY: 60,
      petWidth: 64,
      toY: 150
    })

    expect(surface).toMatchObject({ left: 40, right: 280, y: 92 })
    expect(surface!.confidence).toBeGreaterThan(0.9)
  })

  it('detects a single-pixel horizontal rule whose regional average is too weak', () => {
    const frame = bitmap(320, 180, (_x, y) => (y === 92 ? 135 : 100))

    const surface = detectHorizontalContrastSurface(frame, {
      centerX: 160,
      footY: 58,
      fromY: 60,
      petWidth: 64,
      toY: 150
    })

    expect(surface).toMatchObject({ left: 0, right: 320, y: 92 })
  })

  it('detects a low-contrast horizontal boundary as a jump support', () => {
    const frame = bitmap(320, 180, (_x, y) => (y >= 92 ? 116 : 100))

    const surface = detectHorizontalContrastSurface(frame, {
      centerX: 160,
      footY: 58,
      fromY: 60,
      petWidth: 64,
      toY: 150
    })

    expect(surface).toMatchObject({ left: 0, right: 320, y: 92 })
  })

  it('detects a faint single-pixel rule without relaxing horizontal continuity', () => {
    const frame = bitmap(320, 180, (_x, y) => (y === 92 ? 116 : 100))

    const surface = detectHorizontalContrastSurface(frame, {
      centerX: 160,
      footY: 58,
      fromY: 60,
      petWidth: 64,
      toY: 150
    })

    expect(surface).toMatchObject({ left: 0, right: 320, y: 92 })
  })

  it('reconnects a real edge hidden by the pet around the current foot line', () => {
    const frame = bitmap(320, 180, (x, y) => {
      if (x >= 40 && x < 280 && y >= 72) {
        return x >= 120 && x <= 200 && y < 78 ? 45 : 210
      }

      return 45
    })

    const surface = detectHorizontalContrastSurface(frame, {
      centerX: 160,
      footY: 72,
      fromY: 66,
      petWidth: 64,
      toY: 150
    })

    expect(surface).toMatchObject({ left: 40, right: 280, y: 72 })
  })

  it('rejects a short high-contrast mark such as text or an icon', () => {
    const frame = bitmap(320, 180, (x, y) => (x >= 125 && x < 195 && y >= 92 ? 230 : 40))

    expect(
      detectHorizontalContrastSurface(frame, {
        centerX: 160,
        footY: 58,
        fromY: 60,
        petWidth: 64,
        toY: 150
      })
    ).toBeNull()
  })
})
