import { describe, expect, it } from 'vitest'

import { computeContainBox, displayToNatural } from './image-annotate'

describe('computeContainBox', () => {
  it('scales to fit width when the image is wider than the box', () => {
    const box = computeContainBox(400, 300, 1000, 500)

    expect(box.scale).toBeCloseTo(0.4)
    expect(box.width).toBeCloseTo(400)
    expect(box.height).toBeCloseTo(200)
    expect(box.offsetX).toBeCloseTo(0)
    expect(box.offsetY).toBeCloseTo(50) // vertical letterbox
  })

  it('scales to fit height when the image is taller than the box', () => {
    const box = computeContainBox(400, 300, 500, 1000)

    expect(box.scale).toBeCloseTo(0.3)
    expect(box.width).toBeCloseTo(150)
    expect(box.height).toBeCloseTo(300)
    expect(box.offsetX).toBeCloseTo(125) // horizontal letterbox
    expect(box.offsetY).toBeCloseTo(0)
  })

  it('handles exact fits and degenerate input', () => {
    const exact = computeContainBox(200, 100, 200, 100)
    expect(exact.scale).toBe(1)
    expect(exact.offsetX).toBe(0)

    const zero = computeContainBox(0, 100, 200, 100)
    expect(zero.scale).toBe(1) // safe fallback, no NaN
  })
})

describe('displayToNatural', () => {
  it('converts display pixels back to source pixels', () => {
    const box = computeContainBox(400, 300, 1000, 500) // scale 0.4

    const natural = displayToNatural({ x: 40, y: 20 }, box)
    expect(natural.x).toBeCloseTo(100)
    expect(natural.y).toBeCloseTo(50)
  })

  it('is a no-op at scale 1', () => {
    const box = computeContainBox(200, 100, 200, 100)
    expect(displayToNatural({ x: 42, y: 7 }, box)).toEqual({ x: 42, y: 7 })
  })
})
