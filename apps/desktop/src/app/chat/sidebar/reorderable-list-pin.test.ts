import type { Active, ClientRect, DroppableContainer, UniqueIdentifier } from '@dnd-kit/core'
import { describe, expect, it, vi } from 'vitest'

import { createReorderPin, type ReorderSlotResolver } from './reorderable-list'

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

const at = (y: number, pointerCoordinates: null | { x: number; y: number } = { x: 100, y }) => ({
  active: active('a'),
  collisionRect: { bottom: y, height: 0, left: 0, right: 0, top: y, width: 0, x: 0, y },
  droppableContainers: containers(),
  droppableRects: rects(),
  pointerCoordinates
})

/** dnd-kit takes `over` from the FIRST collision only, so that is the whole answer. */
const over = (pin: ReturnType<typeof createReorderPin>, y: number) => {
  const [first] = pin.detect(at(y))

  return first ? String(first.id) : null
}

/** The same, for a drag with no pointer at all (the keyboard sensor). */
const overKeyboard = (pin: ReturnType<typeof createReorderPin>, y: number) => {
  const [first] = pin.detect(at(y, null))

  return first ? String(first.id) : null
}

describe('the reorder pin', () => {
  it("takes the policy's answer as the drop target", () => {
    // The policy decides which row the dragged item has taken the place of; the list's reflow follows
    // it exactly, because `over` is what the sorting strategy displaces rows for.
    const slot = vi.fn<ReorderSlotResolver>(() => 'c')

    expect(over(createReorderPin(slot), 60)).toBe('c')
    expect(slot).toHaveBeenCalledWith({ x: 100, y: 60 })
  })

  it('holds the dragged item itself while the policy says nothing has moved', () => {
    const pin = createReorderPin(() => null)

    // `over` is the active item, which the strategy reads as a zero displacement. An empty collision
    // list would null the dragged row's transform instead — it would snap back into its own slot.
    expect(over(pin, 60)).toBe('a')
    expect(over(pin, 300)).toBe('a')
  })

  it('keeps the sensor pointer each pass saw, for the drag events that cannot report it', () => {
    const pin = createReorderPin(() => null)

    pin.detect(at(60))

    // `pointerCoordinates` is the activation point plus the pointer's OWN movement. A drag event's
    // `delta` is the scroll-adjusted one instead, so a list that auto-scrolls under a still pointer
    // would place its chip and frame off the pointer by however far it had scrolled.
    expect(pin.pointer).toEqual({ x: 100, y: 60 })

    // A keyboard pass must not leave the previous drag's pointer behind for the next one.
    pin.detect(at(60, null))

    expect(pin.pointer).toBeNull()
  })

  it('leaves a keyboard drag to the list — no pointer, nothing to resolve', () => {
    const slot = vi.fn<ReorderSlotResolver>(() => 'c')

    expect(overKeyboard(createReorderPin(slot), 60)).toBe('a')
    expect(slot).not.toHaveBeenCalled()
  })

  it('passes everything through when the list has no policy', () => {
    // A plain reorderable list keeps dnd-kit's own behaviour: the nearest centre wins.
    expect(over(createReorderPin(undefined), 60)).toBe('b')
    expect(over(createReorderPin(undefined), topOf(2) + 5)).toBe('c')
  })
})
