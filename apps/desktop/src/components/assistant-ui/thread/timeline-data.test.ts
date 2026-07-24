import { describe, expect, it } from 'vitest'

import { activeTimelineIndex, deriveTimelineEntries, timelinePreview } from './timeline-data'

describe('timelinePreview', () => {
  it('collapses whitespace to a single line', () => {
    expect(timelinePreview('hello\n\n  world\tagain')).toBe('hello world again')
  })

  it('truncates with an ellipsis past the limit', () => {
    const out = timelinePreview('abcdefghij', 5)
    expect(out).toBe('abcd…')
    expect(out.length).toBe(5)
  })
})

describe('deriveTimelineEntries', () => {
  it('keeps non-empty user prompts in order', () => {
    expect(
      deriveTimelineEntries([
        { id: 'u1', role: 'user', text: 'first' },
        { id: 'a1', role: 'assistant', text: 'answer' },
        { id: 'u2', role: 'user', text: '  second  ' }
      ])
    ).toEqual([
      { id: 'u1', preview: 'first' },
      { id: 'u2', preview: 'second' }
    ])
  })

  it('drops blanks and background-process notifications', () => {
    expect(
      deriveTimelineEntries([
        { id: 'u1', role: 'user', text: '   ' },
        { id: 'u2', role: 'user', text: '[IMPORTANT: Background process 123 finished]' },
        { id: 'u3', role: 'user', text: 'real prompt' }
      ]).map(e => e.id)
    ).toEqual(['u3'])
  })
})

describe('activeTimelineIndex', () => {
  it('returns the last prompt scrolled to or above the top edge', () => {
    expect(activeTimelineIndex([-400, -10, 320])).toBe(1)
  })

  it('falls back to the first rendered entry', () => {
    expect(activeTimelineIndex([null, 120, 480])).toBe(1)
    expect(activeTimelineIndex([null, null])).toBe(0)
  })

  it('highlights the last entry when scrolled to the bottom', () => {
    // Messages 0-1 are above the viewport, 2-3 are visible inside it.
    // User is at the bottom (isAtBottom=true). Last entry (index 3) wins.
    expect(activeTimelineIndex([-400, -10, 150, 450], 8, 800, true)).toBe(3)
  })

  it('preserves existing behaviour when viewportHeight is 0 (backward compatible)', () => {
    // Same offsets but viewportHeight=0 — shortcut doesn't fire.
    expect(activeTimelineIndex([-400, -10, 150, 450], 8, 0)).toBe(1)
  })

  it('does not highlight last entry when not at bottom (short thread, all visible)', () => {
    // All messages visible in a tall viewport, but user is NOT at the bottom
    // (e.g. short thread, or scrolled to top). Falls through to mid-scroll.
    expect(activeTimelineIndex([50, 180, 310, 440, 570], 8, 1000, false)).toBe(0)
  })

  it('does not highlight last entry when it is scrolled below the viewport', () => {
    // Last entry is 1000px below the top, viewport only 800px tall —
    // it is offscreen. Even at bottom, shortcut does not fire.
    expect(activeTimelineIndex([-400, -10, 320, 1000], 8, 800, true)).toBe(1)
  })
})
