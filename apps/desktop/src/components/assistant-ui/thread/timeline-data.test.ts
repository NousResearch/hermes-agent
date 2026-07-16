import { describe, expect, it } from 'vitest'

import { activeTimelineIndex, deriveTimelineEntries, timelinePreview, userMessageIndex } from './timeline-data'

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
})

describe('userMessageIndex', () => {
  it('counts user messages until the target id is found', () => {
    expect(
      userMessageIndex(
        [
          { id: 'u1', role: 'user' },
          { id: 'a1', role: 'assistant' },
          { id: 'u2', role: 'user' },
          { id: 'u3', role: 'user' }
        ],
        'u3'
      )
    ).toBe(2)
  })

  it('returns the first user ordinal for the leading user message', () => {
    expect(
      userMessageIndex(
        [
          { id: 'u1', role: 'user' },
          { id: 'a1', role: 'assistant' }
        ],
        'u1'
      )
    ).toBe(0)
  })

  it('returns -1 when the id is not a user message', () => {
    expect(
      userMessageIndex(
        [
          { id: 'u1', role: 'user' },
          { id: 'a1', role: 'assistant' }
        ],
        'a1'
      )
    ).toBe(-1)
  })

  it('returns -1 when the id is missing entirely', () => {
    expect(
      userMessageIndex(
        [
          { id: 'u1', role: 'user' }
        ],
        'unknown'
      )
    ).toBe(-1)
  })

  // Regression for the bridge-resolution asymmetry that produced the original
  // scrollToPrompt mis-targeting bug (teknium1 review, 2026-07-15):
  // `entries.findIndex` returns the position in the FILTERED array (no blanks,
  // no process notifications) — that index is NOT the same as the target's
  // position in the unfiltered `groups` array. The bridge carries the id
  // through and resolves it here against the full message stream so a filtered
  // user message preceding the click target lands on the right slice.
  it('returns the unfiltered ordinal when a blank user message precedes the target', () => {
    // Same fixture as deriveTimelineEntries' "drops blanks" case. The rail's
    // filtered entries array would only contain u3 at index 0; the groups
    // array (built from every message) holds u1, u2, u3 at indices 0, 1, 2.
    expect(
      userMessageIndex(
        [
          { id: 'u1', role: 'user' },
          { id: 'u2', role: 'user' }, // blank in deriveTimelineEntries — would be dropped
          { id: 'u3', role: 'user' }
        ],
        'u3'
      )
    ).toBe(2)
  })

  it('returns the unfiltered ordinal when a process notification precedes the target', () => {
    // The exact case the maintainer asked to lock in: a background-process
    // notification user message between two real prompts must not shift the
    // resolved index away from the click target.
    expect(
      userMessageIndex(
        [
          { id: 'real1', role: 'user' },
          { id: 'proc', role: 'user' }, // matches PROCESS_NOTIFICATION_RE — filtered from `entries`
          { id: 'real2', role: 'user' }
        ],
        'real2'
      )
    ).toBe(2)
  })

  it('matches the gap between filtered entries index and groups index', () => {
    // The bug: rail passed `entries.findIndex(...)` as `targetFirstVisible`.
    // For a session with N filtered-out messages between entries[0] and the
    // target, that index is N short of the correct groups index. Pin the
    // relationship so any future helper has to respect it.
    const messages = [
      { id: 'u1', role: 'user' },
      { id: 'b1', role: 'user' },
      { id: 'b2', role: 'user' },
      { id: 'u2', role: 'user' },
      { id: 'b3', role: 'user' },
      { id: 'u3', role: 'user' }
    ]

    // Filtered entries (the timeline drops any message whose id starts with 'b'
    // — conceptually blanks / process notifications). `userMessageIndex` sees
    // the full stream including those.
    const filteredIds = messages.filter(m => m.id.startsWith('u'))
    const targetIndexInFiltered = filteredIds.findIndex(m => m.id === 'u3')

    expect(targetIndexInFiltered).toBe(2) // u1@0, u2@1, u3@2 in entries (3 visible)
    expect(userMessageIndex(messages, 'u3')).toBe(5) // u1,b1,b2,u2,b3,u3 → u3 is the 5th user message in raw order
  })
})
