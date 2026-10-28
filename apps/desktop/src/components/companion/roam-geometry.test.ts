import { describe, expect, it } from 'vitest'

import { GROUND_EPS, type Ledge, overlapsX, resolveLedge } from './roam-geometry'

const ledge = (y: number, left = 0, right = 1000): Ledge => ({ left, right, y })

describe('overlapsX', () => {
  it('is true only when the walkable ranges share real width', () => {
    expect(overlapsX(ledge(0, 0, 100), ledge(0, 50, 200))).toBe(true)
    expect(overlapsX(ledge(0, 0, 100), ledge(0, 100, 200))).toBe(false) // touching, not overlapping
    expect(overlapsX(ledge(0, 0, 100), ledge(0, 300, 400))).toBe(false)
  })
})

describe('resolveLedge', () => {
  const floor = ledge(600)
  const shelf = ledge(300, 100, 400)

  it('returns the highest surface at or below the feet under the current x', () => {
    // Standing on the shelf line, under the shelf's x-span ⇒ the shelf.
    const companionH = 100
    const onShelf = resolveLedge([floor, shelf], 200, shelf.y - companionH, companionH)
    expect(onShelf).toBe(shelf)
  })

  it('ignores surfaces the companion is not horizontally over', () => {
    const companionH = 100
    // x=800 is past the shelf ⇒ only the floor qualifies.
    const onFloor = resolveLedge([floor, shelf], 800, floor.y - companionH, companionH)
    expect(onFloor).toBe(floor)
  })

  it('falls back to the floor (ledges[0]) when below everything', () => {
    const companionH = 100
    const below = resolveLedge([floor, shelf], 200, 5000, companionH)
    expect(below).toBe(floor)
  })

  it('counts a surface within GROUND_EPS of the feet as standing on it', () => {
    const companionH = 100
    const justAbove = resolveLedge([floor], 10, floor.y - companionH - GROUND_EPS + 0.5, companionH)
    expect(justAbove).toBe(floor)
  })
})
