import { describe, expect, it } from 'vitest'

import { SIDEBAR_COLLAPSE_MEDIA_QUERY, SIDEBAR_DOCK_MIN_WIDTH_PX } from '../layout-constants'

import { BELOW_WIDE_QUERY, BREAKPOINT_MEDIUM_PX, BREAKPOINT_WIDE_PX, COMPACT_QUERY, widthClassFor } from './breakpoints'

describe('the responsive model', () => {
  it('has three width classes split at the two breakpoints, a breakpoint taking the larger class', () => {
    expect(widthClassFor(320)).toBe('compact')
    expect(widthClassFor(BREAKPOINT_MEDIUM_PX - 1)).toBe('compact')
    expect(widthClassFor(BREAKPOINT_MEDIUM_PX)).toBe('medium')
    expect(widthClassFor(768)).toBe('medium')
    expect(widthClassFor(BREAKPOINT_WIDE_PX - 1)).toBe('medium')
    expect(widthClassFor(BREAKPOINT_WIDE_PX)).toBe('wide')
    expect(widthClassFor(1280)).toBe('wide')
  })

  it('states the same edges as media queries (inclusive max-width, shaved under the breakpoint)', () => {
    const max = (query: string) => Number(/max-width: ([\d.]+)px/.exec(query)?.[1])

    expect(max(COMPACT_QUERY)).toBeLessThan(BREAKPOINT_MEDIUM_PX)
    expect(max(COMPACT_QUERY)).toBeGreaterThan(BREAKPOINT_MEDIUM_PX - 1)
    expect(max(BELOW_WIDE_QUERY)).toBeLessThan(BREAKPOINT_WIDE_PX)
    expect(max(BELOW_WIDE_QUERY)).toBeGreaterThan(BREAKPOINT_WIDE_PX - 1)
  })

  it("is classic's sidebar dock point too: one breakpoint for both shells", () => {
    expect(SIDEBAR_DOCK_MIN_WIDTH_PX).toBe(BREAKPOINT_MEDIUM_PX)
    expect(SIDEBAR_COLLAPSE_MEDIA_QUERY).toBe(COMPACT_QUERY)
  })
})
