import { describe, expect, it } from 'vitest'

import { activeTimelineIndex, deriveTimelineEntries, sameTimelineEntries, timelinePreview } from './timeline-data'

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

describe('sameTimelineEntries', () => {
  const rail = [
    { id: 'u1', preview: 'first' },
    { id: 'u2', preview: 'second' }
  ]

  it('treats an identical derivation as unchanged, so the memo can reuse it', () => {
    expect(sameTimelineEntries(rail, [...rail.map(e => ({ ...e }))])).toBe(true)
  })

  it('detects a changed preview, id, or length', () => {
    expect(sameTimelineEntries(rail, [rail[0], { id: 'u2', preview: 'edited' }])).toBe(false)
    expect(sameTimelineEntries(rail, [rail[0], { id: 'u9', preview: 'second' }])).toBe(false)
    expect(sameTimelineEntries(rail, [rail[0]])).toBe(false)
  })

  it('is stable when a filtered-out prompt joins the transcript', () => {
    const withNoise = deriveTimelineEntries([
      { id: 'u1', role: 'user', text: 'first' },
      { id: 'u2', role: 'user', text: 'second' },
      { id: 'u3', role: 'user', text: '[IMPORTANT: Background process 7 finished]' }
    ])

    expect(sameTimelineEntries(rail, withNoise)).toBe(true)
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
})
