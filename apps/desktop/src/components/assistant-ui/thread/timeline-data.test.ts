import { describe, expect, it } from 'vitest'

import {
  activeTimelineIndex,
  deriveTimelineEntries,
  numberTimelineEntries,
  sameTimelineEntries,
  timelineAuthor,
  timelineBarWidth,
  type TimelineEntry,
  timelinePreview
} from './timeline-data'

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

describe('timelineBarWidth', () => {
  it('keeps active maximal and uses the same curve for hover', () => {
    expect(timelineBarWidth(10, 10, 50)).toBe(1)
    expect(timelineBarWidth(50, 10, 50)).toBe(1)
    expect(timelineBarWidth(11, 10, null)).toBe(timelineBarWidth(51, 10, 50))
  })

  it('does not change bars outside the local focus neighborhood', () => {
    expect(timelineBarWidth(30, 10, 50)).toBe(0.5)
    expect(timelineBarWidth(11, 10, null)).toBeGreaterThan(timelineBarWidth(12, 10, null))
    expect(timelineBarWidth(11, 0, 10.5)).toBe(timelineBarWidth(10, 0, 10.5))
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
      { id: 'u1', preview: 'first', author: 'human' },
      { id: 'u2', preview: 'second', author: 'human' }
    ])
  })

  it('keeps injected prompts on the rail, marked system', () => {
    expect(
      deriveTimelineEntries([
        { id: 'u1', role: 'user', text: 'typed' },
        { id: 'u2', role: 'user', text: 'Board updated.', displayKind: 'internal_notification' },
        { id: 'u3', role: 'user', text: '[kanban] Task t_1 completed.' },
        { id: 'u4', role: 'user', text: 'go left', displayKind: 'steer' }
      ]).map(e => [e.id, e.author])
    ).toEqual([
      ['u1', 'human'],
      ['u2', 'system'],
      ['u3', 'system'],
      ['u4', 'human']
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

describe('timelineAuthor', () => {
  it('trusts display_kind first, then falls back to known machine prefixes', () => {
    expect(timelineAuthor('hello')).toBe('human')
    expect(timelineAuthor('[kanban] board updated', 'steer')).toBe('human')
    expect(timelineAuthor('looks typed', 'process_complete')).toBe('system')
    expect(timelineAuthor('[IMPORTANT: You are running as a scheduled cron job. Go.]')).toBe('system')
    expect(timelineAuthor('[Cronjob "nightly" output — done]')).toBe('system')
    expect(timelineAuthor('[IMPORTANT: Background process 7 is still running')).toBe('system')
    expect(timelineAuthor('a [kanban] mention mid-sentence')).toBe('human')
  })
})

describe('numberTimelineEntries', () => {
  const rail: TimelineEntry[] = [
    { id: 'more', preview: 'Show earlier' },
    { id: 'u1', preview: 'one', author: 'human' },
    { id: 's1', preview: 'cron', author: 'system' },
    { id: 'u2', preview: 'two', author: 'human' },
    { id: 'u3', preview: 'three', author: 'human' }
  ]

  it('numbers human prompts only, skipping the placeholder and system entries', () => {
    expect(numberTimelineEntries(rail).map(e => e.number)).toEqual([undefined, 1, undefined, 2, 3])
  })

  it('leaves entries past the known prefix unnumbered instead of guessing across a gap', () => {
    expect(numberTimelineEntries(rail, 3).map(e => e.number)).toEqual([undefined, 1, undefined, undefined, undefined])
    expect(numberTimelineEntries(rail, 0).every(e => e.number === undefined)).toBe(true)
  })

  it('gives the same numbers whatever slice a virtualized rail paints', () => {
    const numbered = numberTimelineEntries(rail)

    expect(numbered.slice(3).map(e => e.number)).toEqual([2, 3])
    expect(numberTimelineEntries(rail)).toEqual(numbered)
  })
})

describe('sameTimelineEntries', () => {
  const rail: TimelineEntry[] = [
    { id: 'u1', preview: 'first', author: 'human' },
    { id: 'u2', preview: 'second', author: 'human' }
  ]

  it('treats an identical derivation as unchanged, so the memo can reuse it', () => {
    expect(sameTimelineEntries(rail, [...rail.map(e => ({ ...e }))])).toBe(true)
  })

  it('detects a changed preview, id, or length', () => {
    expect(sameTimelineEntries(rail, [rail[0], { id: 'u2', preview: 'edited' }])).toBe(false)
    expect(sameTimelineEntries(rail, [rail[0], { id: 'u9', preview: 'second' }])).toBe(false)
    expect(sameTimelineEntries(rail, [rail[0]])).toBe(false)
    expect(sameTimelineEntries(rail, [rail[0], { ...rail[1], author: 'system' }])).toBe(false)
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
