import { describe, expect, it } from 'vitest'

import { computeCaptureRect } from './screenshot'

describe('computeCaptureRect', () => {
  it('pads the element rect by the margin on all sides', () => {
    const rect = computeCaptureRect(
      { height: 32, width: 80, x: 100, y: 200 },
      { pageHeight: 1000, pageWidth: 1280 }
    )

    expect(rect).toEqual({ height: 72, width: 120, x: 80, y: 180 })
  })

  it('clamps to the page bounds (top-left overflow)', () => {
    const rect = computeCaptureRect(
      { height: 40, width: 60, x: 5, y: 8 },
      { pageHeight: 1000, pageWidth: 1280 }
    )

    expect(rect.x).toBe(0)
    expect(rect.y).toBe(0)
    // right/bottom edges keep their padded extent: (5+60+20), (8+40+20)
    expect(rect.width).toBe(85)
    expect(rect.height).toBe(68)
  })

  it('clamps to the page bounds (bottom-right overflow)', () => {
    const rect = computeCaptureRect(
      { height: 100, width: 200, x: 1200, y: 950 },
      { pageHeight: 1000, pageWidth: 1280 }
    )

    expect(rect.x + rect.width).toBeLessThanOrEqual(1280)
    expect(rect.y + rect.height).toBeLessThanOrEqual(1000)
  })

  it('respects a custom padding value', () => {
    const rect = computeCaptureRect(
      { height: 10, width: 10, x: 100, y: 100 },
      { pageHeight: 1000, pageWidth: 1280 },
      8
    )

    expect(rect).toEqual({ height: 26, width: 26, x: 92, y: 92 })
  })

  it('returns integers (capturePage rejects fractional rects)', () => {
    const rect = computeCaptureRect(
      { height: 33.7, width: 79.2, x: 100.5, y: 200.9 },
      { pageHeight: 1000, pageWidth: 1280 }
    )

    for (const value of Object.values(rect)) {
      expect(Number.isInteger(value)).toBe(true)
    }
  })
})
