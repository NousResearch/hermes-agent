import { beforeEach, describe, expect, it } from 'vitest'

import {
  $transcriptTailBySessionId,
  clearTranscriptTailPaging,
  recordTranscriptTail,
  rewindTranscriptTail
} from './transcript-tail'

const page = (count: number, limit = 10) => ({
  messages: Array.from({ length: count }, (_, i) => ({ id: `m${i}` })),
  pagination: { limit, offset: 0 }
}) as never

describe('recordTranscriptTail no-op suppression', () => {
  beforeEach(() => {
    $transcriptTailBySessionId.set({})
  })

  it('does not notify on an identical re-record (#113842)', () => {
    recordTranscriptTail('s1', page(5))

    let notifications = 0

    const unsub = $transcriptTailBySessionId.subscribe(() => {
      notifications += 1
    })

    notifications = 0

    recordTranscriptTail('s1', page(5))
    unsub()

    expect(notifications).toBe(0)
  })

  it('notifies when the tail actually advances', () => {
    recordTranscriptTail('s1', page(5))

    let notifications = 0

    const unsub = $transcriptTailBySessionId.subscribe(() => {
      notifications += 1
    })

    notifications = 0

    recordTranscriptTail('s1', page(7))
    unsub()

    expect(notifications).toBe(1)
  })

  it('keeps a no-op re-recorded entry at the MRU end so it survives the next eviction', () => {
    clearTranscriptTailPaging()
    recordTranscriptTail('active', page(5))

    for (let i = 0; i < 255; i += 1) {
      recordTranscriptTail(`other-${i}`, page(5))
    }

    // Identical re-record: no publish, but it must still count as recent use.
    recordTranscriptTail('active', page(5))
    recordTranscriptTail('newcomer', page(5))

    expect($transcriptTailBySessionId.get().active).toBeDefined()
    expect($transcriptTailBySessionId.get()['other-0']).toBeUndefined()
  })
})

describe('rewindTranscriptTail', () => {
  beforeEach(() => {
    $transcriptTailBySessionId.set({})
  })

  it('decrements the recorded offset by the rows the store released', () => {
    // page(10) records nextOffset 10; releasing 4 of those rows leaves the next
    // older page starting 4 rows earlier, in the backend's own units.
    recordTranscriptTail('s1', page(10))

    expect(rewindTranscriptTail('s1', 4)).toBe(true)
    expect($transcriptTailBySessionId.get().s1).toMatchObject({ nextOffset: 6, possiblyTruncated: true })
  })

  it('keeps a rewind on the same route the tail was hydrated with', () => {
    recordTranscriptTail('s1', page(10), { connectionId: 'c1', profile: 'work' }, { connectionId: 'c1', profile: 'work' })

    expect(rewindTranscriptTail('s1', 4, { connectionId: 'c1', profile: 'work' })).toBe(true)

    const entry = $transcriptTailBySessionId.get()[JSON.stringify(['c1', 'work', 's1'])]

    expect(entry).toMatchObject({ nextOffset: 6, possiblyTruncated: true })
    expect(entry.profile).toEqual({ connectionId: 'c1', profile: 'work' })
  })

  it('never rewinds past the start of the transcript', () => {
    recordTranscriptTail('s1', page(3))

    expect(rewindTranscriptTail('s1', 9)).toBe(true)
    expect($transcriptTailBySessionId.get().s1).toMatchObject({ nextOffset: 0, possiblyTruncated: true })
  })

  it('refuses a rewind that would release nothing', () => {
    recordTranscriptTail('s1', page(10))

    expect(rewindTranscriptTail('s1', 0)).toBe(false)
    expect($transcriptTailBySessionId.get().s1).toMatchObject({ nextOffset: 10 })
  })

  it('refuses to rewind a session with no recorded page route', () => {
    // No entry means no route to fetch a page from: the caller must keep its
    // rows rather than release history nothing can bring back.
    expect(rewindTranscriptTail('unknown', 4)).toBe(false)
    expect($transcriptTailBySessionId.get()).toEqual({})
  })

  it('refuses an ambiguous rewind when the session has several owner scopes', () => {
    recordTranscriptTail('s1', page(10), { connectionId: 'c1', profile: 'work' }, { connectionId: 'c1', profile: 'work' })
    recordTranscriptTail('s1', page(10), { connectionId: 'c2', profile: 'work' }, { connectionId: 'c2', profile: 'work' })

    expect(rewindTranscriptTail('s1', 4)).toBe(false)
  })
})
