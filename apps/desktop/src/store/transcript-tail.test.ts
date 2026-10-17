import { beforeEach, describe, expect, it } from 'vitest'

import {
  $transcriptTailBySessionId,
  clearTranscriptTailPaging,
  recordTranscriptTail,
  rewindTranscriptTail,
  transcriptTailState
} from './transcript-tail'

const page = (count: number, limit = 10) =>
  ({
    messages: Array.from({ length: count }, (_, i) => ({ id: `m${i}` })),
    pagination: { limit, offset: 0, order: 'latest' as const }
  }) as never

/** A page from a backend that predates the `order` param: it dropped the
 *  unknown query param, answered from the OLDEST row, and still returned a
 *  `pagination` object — without the honoured-order echo. */
const orderlessPage = (count: number, limit = 10) =>
  ({
    messages: Array.from({ length: count }, (_, i) => ({ id: `m${i}` })),
    pagination: { limit, offset: 0, returned: count }
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
    recordTranscriptTail(
      's1',
      page(10),
      { connectionId: 'c1', profile: 'work' },
      { connectionId: 'c1', profile: 'work' }
    )

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
    recordTranscriptTail(
      's1',
      page(10),
      { connectionId: 'c1', profile: 'work' },
      { connectionId: 'c1', profile: 'work' }
    )
    recordTranscriptTail(
      's1',
      page(10),
      { connectionId: 'c2', profile: 'work' },
      { connectionId: 'c2', profile: 'work' }
    )

    expect(rewindTranscriptTail('s1', 4)).toBe(false)
  })
})

describe('recordTranscriptTail with an empty page', () => {
  beforeEach(() => {
    clearTranscriptTailPaging()
  })

  // The REST helper records the tail before the active refresh decides whether
  // the page is authoritative. A transient zero-row read must not turn a
  // known-truncated tail into "nothing earlier to show".
  it('keeps an existing truncated entry so "Show earlier" stays armed', () => {
    recordTranscriptTail('s1', page(10))

    recordTranscriptTail('s1', page(0))

    expect(transcriptTailState('s1')).toMatchObject({ nextOffset: 10, possiblyTruncated: true })
  })
})

describe('order-echo guard (#92508)', () => {
  beforeEach(() => {
    clearTranscriptTailPaging()
  })

  it('never adopts an orderless page as a truncated tail', () => {
    recordTranscriptTail('s1', orderlessPage(10))

    // The rows are the transcript's OLDEST page, so nothing may be counted
    // back from them and no backfill may arm.
    expect(transcriptTailState('s1')).toMatchObject({ nextOffset: 10, possiblyTruncated: false })
  })

  it("never adopts a page stamped with the order it really served ('oldest')", () => {
    recordTranscriptTail('s1', {
      messages: Array.from({ length: 10 }, (_, i) => ({ id: `m${i}` })),
      pagination: { limit: 10, offset: 0, order: 'oldest', returned: 10 }
    } as never)

    expect(transcriptTailState('s1')).toMatchObject({ nextOffset: 10, possiblyTruncated: false })
  })

  it('still arms a real tail when the page echoes order=latest', () => {
    recordTranscriptTail('s1', page(10))

    expect(transcriptTailState('s1')).toMatchObject({ nextOffset: 10, possiblyTruncated: true })
  })
})

// #133569: every `getLatestSessionMessages` re-records the tail, and a fresh
// hydration read always starts at the NEWEST row. Adopting it wholesale drags
// a tail "Show earlier" already paged back past down to the offset-0 page: each
// further click then re-fetches rows the store already holds, the merge is an
// identity, and the button never retires. Paging state may only advance here.
describe('recordTranscriptTail does not regress paging progress (#133569)', () => {
  beforeEach(() => {
    $transcriptTailBySessionId.set({})
  })

  /** A full page whose rows were counted back from a later offset. */
  const advancedPage = (offset: number, count = 10) =>
    ({
      messages: Array.from({ length: count }, (_, i) => ({ id: `m${i}` })),
      pagination: { limit: 10, offset, order: 'latest' as const }
    }) as never

  it('keeps a further-along offset, and adopts the incoming route, when hydration re-records', () => {
    const owner = { connectionId: 'c1', profile: 'work' }

    recordTranscriptTail('s1', advancedPage(30), owner, owner)
    expect(transcriptTailState('s1')).toMatchObject({ nextOffset: 40, possiblyTruncated: true })

    // Background hydration of the same session, now reached over a re-bound
    // route: newest page, offset 0, full. Paging state must not move, but the
    // route the next older-page fetch is sent with must follow the connection
    // that actually answered.
    recordTranscriptTail('s1', page(10), { connectionId: 'c2', profile: 'work' }, owner)

    expect(transcriptTailState('s1')).toMatchObject({ nextOffset: 40, possiblyTruncated: true })
    expect(transcriptTailState('s1')?.profile).toEqual({ connectionId: 'c2', profile: 'work' })
  })

  it('still retires the offer when a fresh page proves the tail complete', () => {
    recordTranscriptTail('s1', advancedPage(30))
    expect(transcriptTailState('s1')).toMatchObject({ possiblyTruncated: true })

    // Short newest page: the backend has fewer rows than one page, so nothing
    // older exists and "Show earlier" must stand down.
    recordTranscriptTail('s1', page(6))

    expect(transcriptTailState('s1')?.possiblyTruncated).toBe(false)
  })

  it('does not re-arm a tail that already reported everything loaded', () => {
    // The session paged all the way back: nextOffset is past the last page and
    // possiblyTruncated is false, so the button has retired.
    recordTranscriptTail('s1', advancedPage(278, 8))
    expect(transcriptTailState('s1')).toMatchObject({ nextOffset: 286, possiblyTruncated: false })

    // A later hydration read of the same session always comes back full
    // (limit 10). It must not resurrect the offer — only `rewindTranscriptTail`
    // (transcript retention) is allowed to re-arm it.
    recordTranscriptTail('s1', page(10))

    expect(transcriptTailState('s1')).toMatchObject({ nextOffset: 286, possiblyTruncated: false })
  })

  it('re-arms a previously complete tail that the session outgrew', () => {
    // The whole transcript was one short page: 8 rows, nothing older.
    recordTranscriptTail('s1', advancedPage(0, 8))
    expect(transcriptTailState('s1')).toMatchObject({ nextOffset: 8, possiblyTruncated: false })

    // The session kept going and now spans more than a page. The fresh read
    // reaches FURTHER than the recorded state, so it is progress, not a
    // regression: rows nobody has loaded exist and the offer must come back.
    recordTranscriptTail('s1', page(10))

    expect(transcriptTailState('s1')).toMatchObject({ nextOffset: 10, possiblyTruncated: true })
  })
})
