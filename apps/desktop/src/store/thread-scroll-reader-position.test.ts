import { describe, expect, it } from 'vitest'

import { shouldReapplyFrozenThreadScrollOffset, threadScrollTargetTop } from './thread-scroll'

// A reader parked 800 px above the bottom of the transcript.
const READER = { fromBottom: 800, kind: 'offset' as const }

const metrics = (scrollHeight: number, anchorTop?: number) => ({
  clearanceHeight: 120,
  clientHeight: 600,
  scrollHeight,
  ...(anchorTop === undefined ? {} : { anchorTop })
})

/**
 * The reported defect: while a reader is scrolled up, a turn finishes, the final
 * assistant message lands *below* the viewport, and the viewport is dragged down
 * with the growth — the reader loses their place ("had to scroll 3 km back").
 *
 * Arithmetic of the drag, on the old height-only decision:
 *   before: target top = 5000 - 600 - 800 = 3600
 *   after 600px of growth below: 5600 - 600 - 800 = 4200   -> +600 px downward
 * Re-applying a fromBottom offset follows content that grows BELOW the viewport,
 * which is exactly what a reader must not get.
 */
describe('a reader scrolled up keeps their place when the turn grows below', () => {
  it('characterizes the drag: re-applying the offset moves the viewport down by the growth', () => {
    const before = metrics(5000, 200)
    const after = metrics(5600, 200)

    expect(threadScrollTargetTop(READER, before)).toBe(3600)
    expect(threadScrollTargetTop(READER, after)).toBe(4200)
  })

  it('does not re-apply a reader offset for growth below the viewport', () => {
    expect(shouldReapplyFrozenThreadScrollOffset(READER, true, metrics(5000, 200), metrics(5600, 200))).toBe(false)
  })

  it('still re-applies for growth above the reader, so prepend keeps the same content on screen', () => {
    expect(shouldReapplyFrozenThreadScrollOffset(READER, true, metrics(5000, 200), metrics(5600, 800))).toBe(true)
  })

  it('keeps the height-only behaviour when no anchor row could be measured', () => {
    expect(shouldReapplyFrozenThreadScrollOffset(READER, true, metrics(5000), metrics(5600))).toBe(true)
    expect(shouldReapplyFrozenThreadScrollOffset(READER, true, metrics(5000), metrics(5000))).toBe(false)
  })

  it('still never re-applies while the settle loop is running', () => {
    expect(shouldReapplyFrozenThreadScrollOffset(READER, false, metrics(5000, 200), metrics(5600, 800))).toBe(false)
  })
})
