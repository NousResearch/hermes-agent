import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { $transcriptTailBySessionId, recordTranscriptTail, transcriptTailState } from '@/store/transcript-tail'

import {
  _resetTranscriptBackfillForTests,
  backfillOlderTranscriptPage,
  extendRefreshPageToOverlap,
  graftRefreshedTailOntoBackfill,
  mergeOlderTranscriptPage,
  transcriptBackfillAvailable
} from './transcript-backfill'

vi.mock('@/hermes', () => ({
  getOlderSessionMessages: vi.fn()
}))

const { getOlderSessionMessages } = await import('@/hermes')

const chat = (id: string, rowId?: number): ChatMessage => ({
  id,
  role: 'user',
  parts: [{ type: 'text', text: id }],
  ...(rowId !== undefined ? { rowId } : {})
})

// A stored SessionMessage row: distinct timestamps keep toChatMessages ids
// unique and the row id survives as ChatMessage.rowId.
const row = (rowId: number, text: string) => ({
  id: rowId,
  role: 'user' as const,
  content: text,
  timestamp: 1_000 + rowId
})

describe('transcript tail bookkeeping', () => {
  beforeEach(() => {
    $transcriptTailBySessionId.set({})
  })

  it('marks a full page as possibly truncated with the next offset', () => {
    recordTranscriptTail(
      'stored-1',
      {
        messages: Array.from({ length: 120 }, (_, index) => row(index + 500, `m${index}`)),
        pagination: { limit: 120, offset: 0, order: 'latest', returned: 120 }
      },
      'work'
    )

    expect(transcriptTailState('stored-1')).toEqual({ nextOffset: 120, possiblyTruncated: true, profile: 'work' })
    expect(transcriptBackfillAvailable('stored-1')).toBe(true)
  })

  it('marks a short page as complete', () => {
    recordTranscriptTail('stored-1', {
      messages: [row(1, 'only')],
      pagination: { limit: 120, offset: 0, order: 'latest', returned: 1 }
    })

    expect(transcriptBackfillAvailable('stored-1')).toBe(false)
  })

  it('treats a legacy response without pagination metadata as complete', () => {
    recordTranscriptTail('stored-1', {
      messages: Array.from({ length: 700 }, (_, index) => row(index, `m${index}`))
    })

    expect(transcriptBackfillAvailable('stored-1')).toBe(false)
  })

  it('counts the same session id on separate connections as separate entries', () => {
    const page = {
      messages: [row(1, 'tail')],
      pagination: { limit: 120, offset: 0, order: 'latest' as const, returned: 1 }
    }

    const sourceA = { connectionId: 'source-a', profile: 'backend' }
    const sourceB = { connectionId: 'source-b', profile: 'backend' }

    recordTranscriptTail('same-session', page, sourceA)
    recordTranscriptTail('same-session', page, sourceB)

    expect(Object.keys($transcriptTailBySessionId.get())).toHaveLength(2)
    expect(transcriptTailState('same-session', sourceA)?.profile).toEqual(sourceA)
    expect(transcriptTailState('same-session', sourceB)?.profile).toEqual(sourceB)
    expect(transcriptTailState('same-session')).toBeUndefined()
  })

  it('bounds entries and deterministically evicts the oldest scoped identity', () => {
    const page = {
      messages: [row(1, 'tail')],
      pagination: { limit: 120, offset: 0, order: 'latest' as const, returned: 1 }
    }

    const scope = { connectionId: 'source-a', profile: 'backend' }

    for (let index = 0; index < 257; index += 1) {
      recordTranscriptTail(`bounded-${index}`, page, scope)
    }

    expect(Object.keys($transcriptTailBySessionId.get())).toHaveLength(256)
    expect(transcriptTailState('bounded-0', scope)).toBeUndefined()
    expect(transcriptTailState('bounded-1', scope)).toBeDefined()
    expect(transcriptTailState('bounded-256', scope)).toBeDefined()
  })
})

describe('mergeOlderTranscriptPage', () => {
  it('prepends the older page and preserves chronological order', () => {
    const existing = [chat('c', 3), chat('d', 4)]
    const older = [chat('a', 1), chat('b', 2)]

    expect(mergeOlderTranscriptPage(existing, older).map(m => m.id)).toEqual(['a', 'b', 'c', 'd'])
  })

  it('dedupes rows the store already holds by durable row id', () => {
    const existing = [chat('b', 2), chat('c', 3)]
    // Offset drift: the fetched page overlaps one row we already have.
    const older = [chat('a', 1), chat('b-refetched', 2)]

    expect(mergeOlderTranscriptPage(existing, older).map(m => m.rowId)).toEqual([1, 2, 3])
  })

  it('keeps newer rows after an old tail when a drifting offset returns both overlap and subsequent turns', () => {
    // A page initially ending at row 6 was cached. New turns persisted before
    // the older-page request, so its offset now lands across rows 4–8.
    const existing = [
      chat('one', 1),
      chat('two', 2),
      chat('three', 3),
      chat('four', 4),
      chat('five', 5),
      chat('six', 6)
    ]

    const fetched = [
      chat('four-refetched', 4),
      chat('five-refetched', 5),
      chat('six-refetched', 6),
      chat('seven', 7),
      chat('eight', 8)
    ]

    const merged = mergeOlderTranscriptPage(existing, fetched)

    expect(merged.map(message => message.rowId)).toEqual([1, 2, 3, 4, 5, 6, 7, 8])
    expect(merged.slice(0, 6)).toEqual(existing)
    expect(mergeOlderTranscriptPage(merged, fetched)).toBe(merged)
  })

  it('keeps reference identity when every older row is already present', () => {
    const existing = [chat('a', 1), chat('b', 2)]
    const older = [chat('a', 1)]

    expect(mergeOlderTranscriptPage(existing, older)).toBe(existing)
  })

  it('refuses to paint an older page as the whole transcript', () => {
    const existing: ChatMessage[] = []

    expect(mergeOlderTranscriptPage(existing, [chat('a', 1)])).toBe(existing)
  })
})

describe('graftRefreshedTailOntoBackfill', () => {
  it('keeps the backfilled prefix when the refreshed tail anchors inside it', () => {
    const previous = [chat('a', 1), chat('b', 2), chat('c', 3)]
    const refreshed = [chat('b', 2), chat('c', 3), chat('d', 4)]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous).map(m => m.rowId)).toEqual([1, 2, 3, 4])
  })

  it('returns the refreshed tail unchanged when no anchor is found', () => {
    const previous = [chat('x', 90), chat('y', 91), chat('z', 92)]
    const refreshed = [chat('p', 200), chat('q', 201)]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous)).toBe(refreshed)
  })

  it('moves an older row that was glued on after the tail back to stored order', () => {
    const previous = [chat('tail-a', 700), chat('tail-b', 701), chat('tail-c', 746), chat('kcsie', 662)]
    const refreshed = [chat('kcsie', 662), chat('tail-a', 700), chat('tail-b', 701), chat('tail-c', 746)]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous).map(message => message.rowId)).toEqual([
      662, 700, 701, 746
    ])
  })

  it('keeps the fresh page copy when the same stored id is on both sides', () => {
    const previous = [chat('tail-a', 700), chat('stale-b', 701), chat('tail-c', 746)]
    const refreshed = [chat('kcsie', 662), chat('fresh-b', 701), chat('tail-c', 746)]

    const merged = graftRefreshedTailOntoBackfill(refreshed, previous)

    expect(merged.map(message => message.rowId)).toEqual([662, 700, 701, 746])
    expect(merged.find(message => message.rowId === 701)).toBe(refreshed[1])
  })

  it('keeps the earlier transcript when a page-local fold precedes a shared durable row', () => {
    const previous = [chat('earlier', 1), chat('prompt', 2), chat('reply', 3)]
    const refreshed = [chat('page-local-fold'), chat('reply-refetched', 3), chat('new-reply', 4)]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous).map(m => m.rowId)).toEqual([1, 2, undefined, 3, 4])
  })

  it('returns the refreshed tail when it is not shorter than the previous transcript', () => {
    const previous = [chat('a', 1)]
    const refreshed = [chat('a', 1), chat('b', 2)]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous)).toBe(refreshed)
  })
})

describe('extendRefreshPageToOverlap', () => {
  it('reads older pages until a long refresh shares a durable row with the rendered transcript', async () => {
    const previous = [chat('earlier', 1), chat('prompt', 2), chat('reply', 3)]
    const readOlderPage = vi.fn().mockResolvedValueOnce([chat('reply-refetched', 3), chat('tool-fold', 4)])

    const extended = await extendRefreshPageToOverlap(
      [chat('new-tool', 5), chat('new-reply', 6)],
      previous,
      readOlderPage
    )

    expect(readOlderPage).toHaveBeenCalledTimes(1)
    expect(graftRefreshedTailOntoBackfill(extended, previous).map(message => message.rowId)).toEqual([1, 2, 3, 4, 5, 6])
  })

})

describe('backfillOlderTranscriptPage', () => {
  beforeEach(() => {
    $transcriptTailBySessionId.set({})
    _resetTranscriptBackfillForTests()
    vi.mocked(getOlderSessionMessages).mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  const truncatedTail = (nextOffset = 120) => {
    recordTranscriptTail('stored-1', {
      messages: Array.from({ length: 120 }, (_, index) => row(index + nextOffset, `tail${index}`)),
      pagination: { limit: 120, offset: 0, order: 'latest', returned: 120 }
    })
  }

  it('fetches the recorded next offset and applies the converted page', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(1, 'older-1'), row(2, 'older-2')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 2 },
      session_id: 'stored-1'
    } as never)

    const applyOlderPage = vi.fn()

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage
    })

    expect(applied).toBe(true)
    expect(getOlderSessionMessages).toHaveBeenCalledWith('stored-1', undefined, 120)
    expect(applyOlderPage).toHaveBeenCalledTimes(1)
    expect(applyOlderPage.mock.calls[0][0].map((m: ChatMessage) => m.rowId)).toEqual([1, 2])
    // A short older page means the transcript is now fully loaded.
    expect(transcriptBackfillAvailable('stored-1')).toBe(false)
  })

  it('keeps the live tail in order when the fetched offset page includes subsequently persisted rows', async () => {
    recordTranscriptTail('stored-1', {
      messages: [row(4, 'four'), row(5, 'five'), row(6, 'six')],
      pagination: { limit: 3, offset: 0, order: 'latest', returned: 3 }
    })
    // Four rows persisted since hydration. Offset 3 now selects rows 5–7,
    // rather than a page wholly before the cached 4–6 tail.
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(5, 'five'), row(6, 'six'), row(7, 'seven')],
      pagination: { limit: 3, offset: 3, order: 'latest', returned: 3 },
      session_id: 'stored-1'
    } as never)

    let visible = [chat('four', 4), chat('five', 5), chat('six', 6)]

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: page => {
        visible = mergeOlderTranscriptPage(visible, page)
      }
    })

    expect(applied).toBe(true)
    expect(getOlderSessionMessages).toHaveBeenCalledWith('stored-1', undefined, 3)
    expect(visible.map(message => message.rowId)).toEqual([4, 5, 6, 7])
    expect(transcriptTailState('stored-1')?.nextOffset).toBe(6)

    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(2, 'two'), row(3, 'three'), row(4, 'four')],
      pagination: { limit: 3, offset: 6, order: 'latest', returned: 3 },
      session_id: 'stored-1'
    } as never)

    await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: page => {
        visible = mergeOlderTranscriptPage(visible, page)
      }
    })

    expect(getOlderSessionMessages).toHaveBeenLastCalledWith('stored-1', undefined, 6)
    expect(visible.map(message => message.rowId)).toEqual([2, 3, 4, 5, 6, 7])
  })

  it('backfills the matching connection when two owners share one session id', async () => {
    const sourceA = { connectionId: 'source-a', profile: 'backend-a' }
    const sourceB = { connectionId: 'source-b', profile: 'backend-b' }

    const page = {
      messages: Array.from({ length: 120 }, (_, index) => row(index, `tail${index}`)),
      pagination: { limit: 120, offset: 0, order: 'latest' as const, returned: 120 }
    }

    recordTranscriptTail('same-session', page, sourceA)
    recordTranscriptTail('same-session', page, sourceB)
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(1, 'older')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'same-session'
    } as never)

    await backfillOlderTranscriptPage({
      storedSessionId: 'same-session',
      profile: sourceB,
      isCurrent: () => true,
      applyOlderPage: vi.fn()
    })

    expect(getOlderSessionMessages).toHaveBeenCalledWith('same-session', sourceB, 120)
    expect(transcriptTailState('same-session', sourceA)).toMatchObject({ possiblyTruncated: true })
    expect(transcriptTailState('same-session', sourceB)).toMatchObject({ possiblyTruncated: false })
    expect(transcriptTailState('same-session')).toBeUndefined()
  })

  it('keeps backfill available while pages keep coming back full', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: Array.from({ length: 120 }, (_, index) => row(index, `older${index}`)),
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 120 },
      session_id: 'stored-1'
    } as never)

    await backfillOlderTranscriptPage({ storedSessionId: 'stored-1', isCurrent: () => true, applyOlderPage: vi.fn() })

    expect(transcriptTailState('stored-1')).toMatchObject({ nextOffset: 240, possiblyTruncated: true })
  })

  it('falls back to the full transcript when a legacy backend returns no pagination metadata', async () => {
    truncatedTail()
    // Legacy backend: ignores limit/offset/order and one-shots everything.
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: Array.from({ length: 700 }, (_, index) => row(index, `full${index}`)),
      session_id: 'stored-1'
    } as never)

    const applyOlderPage = vi.fn()

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage
    })

    expect(applied).toBe(true)
    expect(applyOlderPage.mock.calls[0][0]).toHaveLength(700)
    // One-shot full transcript: the REST action retires.
    expect(transcriptBackfillAvailable('stored-1')).toBe(false)
  })

  it('discards a stale response after a session switch', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(1, 'older-1')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'stored-1'
    } as never)

    const applyOlderPage = vi.fn()

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      // The user switched sessions while the page was in flight.
      isCurrent: () => false,
      applyOlderPage
    })

    expect(applied).toBe(false)
    expect(applyOlderPage).not.toHaveBeenCalled()
    // Bookkeeping untouched: the next visit re-records the tail anyway.
    expect(transcriptTailState('stored-1')).toMatchObject({ nextOffset: 120, possiblyTruncated: true })
  })

  it('discards an older page when the same session tail was replaced during the fetch', async () => {
    truncatedTail()
    let resolvePage!: (value: unknown) => void
    vi.mocked(getOlderSessionMessages).mockReturnValue(
      // SAFETY: this controlled promise resolves with the exact RPC response shape below.
      new Promise(resolve => {
        resolvePage = resolve
      }) as never
    )
    const applyOlderPage = vi.fn()

    const pending = backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage
    })

    // Rewind/revalidation replaced the display tail without changing the route.
    recordTranscriptTail('stored-1', {
      messages: [row(900, 'replacement')],
      pagination: { limit: 120, offset: 0, order: 'latest', returned: 1 }
    })
    resolvePage({
      messages: [row(1, 'stale older row')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'stored-1'
    })

    expect(await pending).toBe(false)
    expect(applyOlderPage).not.toHaveBeenCalled()
    expect(transcriptTailState('stored-1')).toMatchObject({ nextOffset: 1, possiblyTruncated: false })
  })

  it('shares one in-flight fetch per stored session', async () => {
    truncatedTail()

    let resolvePage: (value: unknown) => void = () => {}

    vi.mocked(getOlderSessionMessages).mockReturnValue(
      // SAFETY: this controlled promise resolves with the exact RPC response shape below.
      new Promise(resolve => {
        resolvePage = resolve
      }) as never
    )

    const first = backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: vi.fn()
    })

    const second = backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: vi.fn()
    })

    expect(second).toBe(first)
    expect(getOlderSessionMessages).toHaveBeenCalledTimes(1)

    resolvePage({
      messages: [row(1, 'older-1')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'stored-1'
    })

    await first
  })

  it('resolves false without fetching when the tail is not truncated', async () => {
    recordTranscriptTail('stored-1', {
      messages: [row(1, 'only')],
      pagination: { limit: 120, offset: 0, order: 'latest', returned: 1 }
    })

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: vi.fn()
    })

    expect(applied).toBe(false)
    expect(getOlderSessionMessages).not.toHaveBeenCalled()
  })

  it('survives a fetch failure and leaves the action retryable', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockRejectedValue(new Error('network down'))

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: vi.fn()
    })

    expect(applied).toBe(false)
    expect(transcriptBackfillAvailable('stored-1')).toBe(true)
  })
})
