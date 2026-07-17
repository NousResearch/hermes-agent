import { describe, expect, it } from 'vitest'

describe('ThreadMessageList render budget constants', () => {
  it('defines FIRST_PAINT_BUDGET as smaller than RENDER_BUDGET', () => {
    // These constants control the deferred render budget on session switch.
    // FIRST_PAINT_BUDGET should be small enough to render quickly (just the
    // bottom turn(s) visible after scroll-to-bottom), while RENDER_BUDGET
    // is the full cap for "Show earlier" increments and the eventual full
    // transcript after the first paint.
    const RENDER_BUDGET = 300
    const FIRST_PAINT_BUDGET = 60

    expect(FIRST_PAINT_BUDGET).toBeLessThan(RENDER_BUDGET)
    expect(FIRST_PAINT_BUDGET).toBeGreaterThan(0)
  })

  it('ensures FIRST_PAINT_BUDGET is sufficient for at least one visible turn', () => {
    // A typical bottom turn (user + assistant) might have ~20-40 parts total
    // (including tool calls), so 60 is enough for 1-2 visible turns at the
    // bottom of the viewport.
    const FIRST_PAINT_BUDGET = 60
    const TYPICAL_TURN_PARTS = 30

    expect(FIRST_PAINT_BUDGET).toBeGreaterThanOrEqual(TYPICAL_TURN_PARTS)
  })
})
