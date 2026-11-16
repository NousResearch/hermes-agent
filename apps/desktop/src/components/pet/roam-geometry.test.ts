import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  blockerLift,
  blockerTopLedge,
  gapX,
  GROUND_EPS,
  hopPlan,
  hopReachable,
  type Ledge,
  overlapsX,
  resolveLedge,
  snapshotLedges,
  splitRuns
} from './roam-geometry'

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
    const petH = 100
    const onShelf = resolveLedge([floor, shelf], 200, shelf.y - petH, petH)
    expect(onShelf).toBe(shelf)
  })

  it('ignores surfaces the pet is not horizontally over', () => {
    const petH = 100
    // x=800 is past the shelf ⇒ only the floor qualifies.
    const onFloor = resolveLedge([floor, shelf], 800, floor.y - petH, petH)
    expect(onFloor).toBe(floor)
  })

  it('falls back to the floor (ledges[0]) when below everything', () => {
    const petH = 100
    const below = resolveLedge([floor, shelf], 200, 5000, petH)
    expect(below).toBe(floor)
  })

  it('counts a surface within GROUND_EPS of the feet as standing on it', () => {
    const petH = 100
    const justAbove = resolveLedge([floor], 10, floor.y - petH - GROUND_EPS + 0.5, petH)
    expect(justAbove).toBe(floor)
  })
})

describe('gapX / hopReachable', () => {
  it('measures the empty span between two ledges', () => {
    expect(gapX(ledge(0, 0, 240), ledge(0, 300, 840))).toBe(60)
    expect(gapX(ledge(0, 0, 400), ledge(0, 300, 840))).toBe(0)
  })

  it('allows a step across a gap no wider than one body width', () => {
    const floorRun = ledge(800, 0, 240)
    const composerTop = ledge(700, 300, 840)

    expect(hopReachable(floorRun, composerTop, 60)).toBe(true) // 60px step, 60px pet
    expect(hopReachable(floorRun, composerTop, 30)).toBe(false) // too far for a smaller pet
  })
})

describe('hopPlan', () => {
  it('springs straight up/down when the spans overlap', () => {
    const from = ledge(800, 0, 500)
    const to = ledge(700, 300, 840)
    const plan = hopPlan(from, to, () => 0.5)

    expect(plan.launch).toBe(400)
    expect(plan.land).toBe(400)
  })

  it('launches from the near edge and lands on the far ledge beside it', () => {
    const floorRun = ledge(800, 0, 240)
    const composerTop = ledge(700, 300, 840)

    expect(hopPlan(floorRun, composerTop)).toEqual({ land: 300, launch: 240 })
    expect(hopPlan(composerTop, floorRun)).toEqual({ land: 240, launch: 300 })
  })
})

describe('splitRuns', () => {
  it('leaves the span untouched when nothing blocks it', () => {
    expect(splitRuns(0, 1000, [], 50)).toEqual([[0, 1000]])
  })

  it('cuts the body footprint of a blocker, leaving the runs beside it', () => {
    // The pet's right edge may touch the blocker's left edge, and its left edge
    // the blocker's right edge — the body itself never overlaps.
    expect(splitRuns(0, 1000, [{ left: 300, right: 700, top: 0 }], 50)).toEqual([
      [0, 250],
      [700, 1000]
    ])
  })

  it('accumulates several blockers', () => {
    expect(
      splitRuns(
        0,
        1000,
        [
          { left: 300, right: 500, top: 0 },
          { left: 600, right: 700, top: 0 }
        ],
        50
      )
    ).toEqual([
      [0, 250],
      [500, 550],
      [700, 1000]
    ])
  })
})

describe('blockerTopLedge', () => {
  it('is the standable span on top of the obstacle', () => {
    expect(blockerTopLedge({ left: 300, right: 900, top: 700 }, 60, 1200)).toEqual({
      left: 300,
      right: 840,
      y: 700
    })
  })

  it('is null when the obstacle is too narrow to stand on', () => {
    expect(blockerTopLedge({ left: 300, right: 340, top: 700 }, 60, 1200)).toBeNull()
  })
})

describe('snapshotLedges / blockerLift on a live DOM', () => {
  const originalWidth = window.innerWidth
  const originalHeight = window.innerHeight

  const setViewport = (width: number, height: number) => {
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: width })
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: height })
  }

  // A composer standing on the window floor: 1200×800 window, box 300→900 at
  // y 700→780 (the bottom is the window edge here — no status bar).
  const mountComposer = (rect: { bottom: number; left: number; right: number; top: number }) => {
    const el = document.createElement('div')
    el.dataset.slot = 'composer-surface'
    document.body.append(el)
    vi.spyOn(el, 'getBoundingClientRect').mockReturnValue({
      bottom: rect.bottom,
      height: rect.bottom - rect.top,
      left: rect.left,
      right: rect.right,
      top: rect.top,
      width: rect.right - rect.left,
      x: rect.left,
      y: rect.top,
      toJSON: () => ({})
    } as DOMRect)

    return el
  }

  afterEach(() => {
    document.body.innerHTML = ''
    vi.restoreAllMocks()
    setViewport(originalWidth, originalHeight)
  })

  it('keeps the floor out of the input box and puts its top edge in instead', () => {
    setViewport(1200, 800)
    mountComposer({ bottom: 780, left: 300, right: 900, top: 700 })

    const ledges = snapshotLedges(60, 68)

    // Two floor runs beside the box, plus the box's own top edge.
    expect(ledges).toEqual([
      { left: 0, right: 240, y: 800 },
      { left: 900, right: 1140, y: 800 },
      { left: 300, right: 840, y: 700 }
    ])

    // Nothing walkable covers the box at floor level.
    for (const candidate of ledges.filter(c => c.y === 800)) {
      expect(candidate.right + 60 > 300 && candidate.left < 900).toBe(false)
    }
  })

  it('drops floor runs too narrow to stand in', () => {
    setViewport(1200, 800)
    // Box almost the full width: the leftover runs are slivers under one pet.
    mountComposer({ bottom: 780, left: 40, right: 1160, top: 700 })

    expect(snapshotLedges(60, 68)).toEqual([{ left: 40, right: 1100, y: 700 }])
  })

  it('lifts a pet standing in front of the box up onto its top edge', () => {
    setViewport(1200, 800)
    mountComposer({ bottom: 780, left: 300, right: 900, top: 700 })

    // Feet on the floor, body over the input box ⇒ climb its top edge.
    expect(blockerLift(60, 68, 500, 800 - 68)).toEqual({ left: 300, right: 840, y: 700 })

    // Already on top of it, or clear of it ⇒ no lift.
    expect(blockerLift(60, 68, 500, 700 - 68)).toBeNull()
    expect(blockerLift(60, 68, 100, 800 - 68)).toBeNull()
  })

  it('ignores a box with no headroom above it', () => {
    setViewport(1200, 800)
    mountComposer({ bottom: 780, left: 300, right: 900, top: 40 })

    expect(snapshotLedges(60, 68)).toEqual([{ left: 0, right: 1140, y: 800 }])
  })
})
