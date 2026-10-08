import type { Active, ClientRect, DroppableContainer, UniqueIdentifier } from '@dnd-kit/core'
import { describe, expect, it } from 'vitest'

import { createReorderPin, type ReorderQuietZone } from './reorderable-list'

/** Four 40px rows, 10px apart, so a GAP is 10px wide between every pair. */
const ROWS = ['a', 'b', 'c', 'd']
const ROW_H = 40
const GAP = 10

const topOf = (index: number) => index * (ROW_H + GAP)

const rects = (): Map<UniqueIdentifier, ClientRect> =>
  new Map(
    ROWS.map((id, index) => {
      const top = topOf(index)

      return [id, { bottom: top + ROW_H, height: ROW_H, left: 0, right: 200, top, width: 200, x: 0, y: top }]
    })
  )

const active = (id: string): Active => ({
  data: { current: {} },
  id,
  rect: { current: { initial: null, translated: null } }
})

const containers = (): DroppableContainer[] =>
  ROWS.map(id => ({
    data: { current: {} },
    disabled: false,
    id,
    key: id,
    node: { current: null },
    rect: { current: null }
  })) as DroppableContainer[]

const at = (y: number) => ({
  active: active('a'),
  collisionRect: { bottom: y, height: 0, left: 0, right: 0, top: y, width: 0, x: 0, y },
  droppableContainers: containers(),
  droppableRects: rects(),
  pointerCoordinates: { x: 100, y }
})

/** dnd-kit takes `over` from the FIRST collision only, so that is the whole answer. */
const over = (pin: ReturnType<typeof createReorderPin>, y: number) => {
  const [first] = pin.detect(at(y))

  return first ? String(first.id) : null
}

/** A policy that claims only the rows — every gap is live, as the projects policy is. */
const rowsOnly: ReorderQuietZone = pointer =>
  pointer !== null && ROWS.some((_, index) => pointer.y >= topOf(index) && pointer.y <= topOf(index) + ROW_H)

const gapAfter = (id: string) => topOf(ROWS.indexOf(id)) + ROW_H + Math.floor(GAP / 2)

/**
 * In the gap below a row, clear of the hold, and closer to the NEXT row's centre than to this one's.
 *
 * `closestCenter` ranks centres, not boxes, so the gap is not where the reorder flips: it flips a few
 * pixels further on, at the midpoint between two centres. Picking a point either side of that midpoint
 * would assert dnd-kit's tie-break, not the pin.
 */
const pastCentreOf = (id: string) => topOf(ROWS.indexOf(id)) + ROW_H + GAP - 3

describe('the reorder pin', () => {
  it('never reports no drop target, so the dragged row keeps its transform', () => {
    const pin = createReorderPin(rowsOnly)

    // Every y on the way down, through the dead centre of each row and each gap.
    for (let y = 0; y <= ROWS.length * (ROW_H + GAP); y += 3) {
      expect(pin.detect(at(y))).not.toEqual([])
    }
  })

  it('holds on the row it was entered from, through the gap it is leaving into', () => {
    const pin = createReorderPin(rowsOnly)

    expect(over(pin, 60)).toBe('b')
    // Inside the gap below b: still held, so the reorder has not resumed inside b's box.
    expect(over(pin, gapAfter('b'))).toBe('b')
    // Past b's bottom edge: released, and the ordinary collision pass takes over.
    expect(over(pin, pastCentreOf('b'))).toBe('c')
  })

  it('crosses a project and back the same way in both directions', () => {
    // Enter c from above, cross its bottom edge, come back up into it, then cross its top edge.
    const down = createReorderPin(rowsOnly)

    expect(over(down, 120)).toBe('c')
    expect(over(down, 145)).toBe('c')
    // Back up INTO c: held again — it did not swap on the way through.
    expect(over(down, 120)).toBe('c')
    // Up past c's top edge, into the gap above it: released, and b is now the nearest centre.
    expect(over(down, 92)).toBe('b')

    // Reaching the same coordinates travelling UP must give the same answers.
    const up = createReorderPin(rowsOnly)

    expect(over(up, 120)).toBe('c')
    expect(over(up, 145)).toBe('c')
    expect(over(up, 120)).toBe('c')
    expect(over(up, 92)).toBe('b')
  })

  it('is per drag: clearing it releases an in-flight hold', () => {
    const pin = createReorderPin(rowsOnly)

    expect(over(pin, 60)).toBe('b')

    pin.clear()

    expect(over(pin, pastCentreOf('b'))).toBe('c')
  })

  it('passes everything through when the list has no policy', () => {
    const pin = createReorderPin(undefined)

    expect(over(pin, 60)).toBe('b')
  })

  it('leaves a keyboard drag alone — no pointer, nothing to claim', () => {
    const pin = createReorderPin(rowsOnly)

    expect(over(pin, 60)).toBe('b')
  })
})