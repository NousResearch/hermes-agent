import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import {
  $transcriptTailBySessionId,
  recordTranscriptTail,
  rewindTranscriptTail,
  transcriptTailState
} from '@/store/transcript-tail'
import type { SessionMessage } from '@/types/hermes'

import {
  _resetTranscriptBackfillForTests,
  backfillOlderTranscriptPage,
  extendRefreshPageToOverlap,
  graftRefreshedTailOntoBackfill,
  mergeOlderTranscriptPage,
  transcriptBackfillAvailable,
  unhideOpeningUserRows
} from './transcript-backfill'

vi.mock('@/hermes', () => ({
  getOlderSessionMessages: vi.fn(),
  getSessionMessages: vi.fn()
}))

const { getOlderSessionMessages, getSessionMessages } = await import('@/hermes')

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

  // Hydration folds a turn into one bubble addressed by its FIRST source row,
  // with the final reply (52203) riding inside as a text part. A drifting
  // older-page fetch that overlaps the just-persisted turn must not paint that
  // reply twice, but must not drop rows the store does not hold either.
  const narration = { type: 'text', text: 'earlier tool narration', sourceRowId: 52201 } as const
  const answer = { type: 'text', text: 'the answer', sourceRowId: 52203 } as const

  const toolCall = {
    type: 'tool-call',
    toolCallId: 't1',
    toolName: 'read_file',
    args: {}
  } as unknown as ChatMessage['parts'][number]

  it.each([
    {
      name: 'drops a fold whose every row is already held live',
      live: { parts: [narration, answer], rowId: 52203 },
      foldParts: [narration, answer],
      expected: ['prompt', 'assistant-stream-1']
    },
    {
      name: 'keeps a fold carrying a tool call and unheld narration',
      live: { parts: [answer], rowId: 52203 },
      foldParts: [toolCall, narration, answer],
      expected: ['prompt', '1770000000000-4-assistant', 'assistant-stream-1']
    },
    {
      name: 'keeps a fold whose first row is held only as a part of another bubble',
      live: { parts: [narration], rowId: 52202 },
      foldParts: [narration, answer],
      expected: ['prompt', '1770000000000-4-assistant', 'assistant-stream-1']
    }
  ])('$name (#123801)', ({ live, foldParts, expected }) => {
    const existing: ChatMessage[] = [
      chat('prompt', 52200),
      { id: 'assistant-stream-1', role: 'assistant', ...live, parts: [...live.parts] }
    ]

    const fold: ChatMessage = {
      id: '1770000000000-4-assistant',
      role: 'assistant',
      parts: [...foldParts],
      rowId: 52201
    }

    expect(
      mergeOlderTranscriptPage(existing, [chat('prompt-refetch', 52200), fold]).map(message => message.id)
    ).toEqual(expected)
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

    // A page that starts mid-turn opens with a tool fold that has no stored
    // id. It stays in front of the stored row it preceded.
    const behindEarlier = [chat('earlier', 500), ...previous]
    const withFold = [chat('fold-tools'), ...refreshed, chat('next', 747)]

    expect(graftRefreshedTailOntoBackfill(withFold, behindEarlier).map(message => message.id)).toEqual([
      'earlier',
      'fold-tools',
      'kcsie',
      'tail-a',
      'tail-b',
      'tail-c',
      'next'
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

  it('does not treat a page that opens on its own orphan tool fold as behind itself (#124311)', () => {
    const page = [chat('orphan-tools'), chat('user', 122886), chat('assistant', 122887)]

    expect(graftRefreshedTailOntoBackfill(page, page)).toBe(page)
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
    vi.mocked(getSessionMessages).mockReset()
    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [], session_id: 'stored-1' } as never)
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
    // The exhausted tail also fetches the origin page (see the #96875 tests
    // below), so assert the older page landed as the FIRST apply.
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

  // #96875: paging can reach the true top while the opening USER turn is still
  // missing — compaction projection stamps the durable opening row
  // display_kind=hidden (hydration drops it) or the last `latest` page begins
  // at the first assistant row. Once the tail bookkeeping reports the session
  // fully loaded, the oldest display rows must be fetched once and the opening
  // user turn prepended so "Show earlier" lands on the greeting.
  it('fetches the oldest page once paging is exhausted and prepends the opening user turn', async () => {
    truncatedTail()
    // The last older page: full paging history ends at the first assistant row.
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [{ ...row(10, 'first-assistant'), role: 'assistant' as const }],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'stored-1'
    } as never)
    // The oldest page (order=oldest, offset 0) holds the greeting, hidden.
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: [
        {
          id: 1,
          role: 'user' as const,
          content: 'opening greeting',
          display_kind: 'hidden',
          timestamp: 1
        },
        { ...row(10, 'first-assistant'), role: 'assistant' as const }
      ],
      pagination: { limit: 20, offset: 0, order: 'oldest', returned: 2 },
      session_id: 'stored-1'
    } as never)

    const applyOlderPage = vi.fn()

    await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage
    })

    expect(getSessionMessages).toHaveBeenCalledWith('stored-1', undefined, {
      includeCompacted: true,
      limit: 20,
      offset: 0,
      order: 'oldest'
    })
    // First call: the regular older page. Second call: the origin page.
    expect(applyOlderPage).toHaveBeenCalledTimes(2)
    const originPage = applyOlderPage.mock.calls[1][0] as ChatMessage[]
    expect(originPage[0].role).toBe('user')
    expect(originPage[0].parts[0]).toMatchObject({ type: 'text', text: 'opening greeting' })
  })

  it('does not fetch the oldest page while older pages remain', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: Array.from({ length: 120 }, (_, index) => row(index, `older${index}`)),
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 120 },
      session_id: 'stored-1'
    } as never)

    await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: vi.fn()
    })

    expect(getSessionMessages).not.toHaveBeenCalled()
  })

  it('skips the origin prepend when the user switched sessions mid-flight', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(10, 'first-assistant')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'stored-1'
    } as never)

    const applyOlderPage = vi.fn()

    await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => false,
      applyOlderPage
    })

    // The stale-response guard discards before any apply; no origin fetch ran.
    expect(applyOlderPage).not.toHaveBeenCalled()
  })

  it('keeps the already-applied older page when the origin fetch fails', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(10, 'first-assistant')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'stored-1'
    } as never)
    vi.mocked(getSessionMessages).mockRejectedValue(new Error('network down'))

    const applyOlderPage = vi.fn()

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage
    })

    expect(applied).toBe(true)
    expect(applyOlderPage).toHaveBeenCalledTimes(1)
  })
})

describe('unhideOpeningUserRows', () => {
  it('clears hidden on the opening user row so hydration keeps the greeting', () => {
    const messages = unhideOpeningUserRows([
      {
        id: 1,
        role: 'user',
        content: 'hello there',
        display_kind: 'hidden',
        timestamp: 1
      }
    ])

    expect(messages[0].display_kind).toBeUndefined()
    expect(messages[0].display_content).toBe('hello there')
  })

  it('leaves a visible opening user row untouched', () => {
    const messages = unhideOpeningUserRows([row(1, 'plain greeting')])

    expect(messages[0].display_kind).toBeUndefined()
    expect(messages[0].display_content).toBeUndefined()
  })

  it('leaves hidden rows after the opening user turn alone', () => {
    const messages = unhideOpeningUserRows([
      row(1, 'greeting'),
      { id: 2, role: 'user', content: 'scaffolding', display_kind: 'hidden', timestamp: 2 }
    ])

    expect(messages[1].display_kind).toBe('hidden')
  })

  it('leaves a hidden opening row without content hidden', () => {
    const messages = unhideOpeningUserRows([{ id: 1, role: 'user', content: '', display_kind: 'hidden', timestamp: 1 }])

    expect(messages[0].display_kind).toBe('hidden')
  })
})

describe('Show earlier paging walks', () => {
  const SID = 'stored-1'
  const LIMIT = 120

  let pagingBackend: number[] = []
  let pagingStore: ChatMessage[] = []

  const pagingRange = (from: number, to: number) => Array.from({ length: to - from + 1 }, (_, i) => from + i)

  /** The backend's `order: 'latest'` page: counted back from the newest row, chronological inside. */
  const pagingLatestPage = (offset: number) => {
    const end = Math.max(0, pagingBackend.length - offset)
    const messages = pagingBackend.slice(Math.max(0, end - LIMIT), end).map(id => row(id, `r${id}`))

    return { messages, pagination: { limit: LIMIT, offset, order: 'latest' as const, returned: messages.length } }
  }

  /** What every refresh and reopen does: record the newest page, graft it onto the held rows. */
  const pagingRefresh = () => {
    const page = pagingLatestPage(0)
    recordTranscriptTail(SID, page)
    pagingStore = graftRefreshedTailOntoBackfill(toChatMessages(page.messages), pagingStore)
  }

  const pagingShowEarlier = () =>
    backfillOlderTranscriptPage({
      storedSessionId: SID,
      isCurrent: () => true,
      heldMessages: () => pagingStore,
      applyOlderPage: older => {
        const merged = mergeOlderTranscriptPage(pagingStore, older)
        const grew = merged !== pagingStore
        pagingStore = merged

        return grew
      }
    })

  const pagingToTop = async () => {
    for (let click = 0; click < 20 && transcriptBackfillAvailable(SID); click++) {
      await pagingShowEarlier()
    }
  }

  const pagingMissing = (expected: number[]) => {
    const held = new Set(pagingStore.map(message => message.rowId))

    return expected.filter(id => !held.has(id))
  }

  beforeEach(() => {
    $transcriptTailBySessionId.set({})
    _resetTranscriptBackfillForTests()
    vi.mocked(getOlderSessionMessages).mockReset()
    // SAFETY: the fake page returns the messages and pagination fields the backfill reads.
    vi.mocked(getOlderSessionMessages).mockImplementation(
      async (_id, _scope, offset) => pagingLatestPage(offset) as never
    )
    vi.mocked(getSessionMessages).mockReset()
    // SAFETY: an empty origin page; the backfill reads only `messages`.
    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [], session_id: SID } as never)
    pagingBackend = pagingRange(1, 600)
    pagingStore = []
  })

  describe('Show earlier across refreshes, /undo, reopen and rewrite', () => {
    it('adds older rows on the first click after a refresh re-records the newest page', async () => {
      pagingRefresh()
      await pagingShowEarlier()
      await pagingShowEarlier()
      const oldestBefore = pagingStore[0].rowId!

      pagingRefresh()
      await pagingShowEarlier()

      expect(pagingStore[0].rowId).toBeLessThan(oldestBefore)
    })

    it('reaches the first row when a refresh lands between every click', async () => {
      pagingRefresh()

      for (let click = 0; click < 10 && transcriptBackfillAvailable(SID); click++) {
        await pagingShowEarlier()
        pagingRefresh()
      }

      expect(pagingMissing(pagingRange(1, 600))).toEqual([])
    })

    it('reaches every row after /undo shortens the transcript while paged back', async () => {
      pagingRefresh()
      await pagingShowEarlier()
      await pagingShowEarlier()

      pagingBackend = pagingRange(1, 590)
      pagingStore = pagingStore.filter(message => message.rowId! <= 590)
      pagingRefresh()
      await pagingToTop()

      expect(pagingMissing(pagingRange(1, 590))).toEqual([])
    })

    it('reaches every row when a released transcript reopens on its newest page', async () => {
      pagingRefresh()
      await pagingShowEarlier()
      await pagingShowEarlier()

      pagingStore = []
      pagingRefresh()
      await pagingToTop()

      expect(pagingMissing(pagingRange(1, 600))).toEqual([])
    })

    it('offers Show earlier again when a fully paged chat reopens on its newest page', async () => {
      pagingBackend = pagingRange(1, 300)
      pagingRefresh()
      await pagingToTop()
      expect(transcriptBackfillAvailable(SID)).toBe(false)

      pagingStore = []
      pagingRefresh()

      expect(transcriptBackfillAvailable(SID)).toBe(true)
    })

    it('retires Show earlier on the next click when a refresh re-arms a fully loaded chat', async () => {
      pagingBackend = pagingRange(1, 300)
      pagingRefresh()
      await pagingToTop()
      pagingRefresh()

      await pagingShowEarlier()

      expect(transcriptBackfillAvailable(SID)).toBe(false)
    })

    it('reaches every row after compaction rewrites the transcript with new row ids', async () => {
      pagingRefresh()
      await pagingShowEarlier()
      await pagingShowEarlier()

      pagingBackend = pagingRange(1001, 1500)
      pagingRefresh()
      await pagingToTop()

      expect(pagingMissing(pagingRange(1001, 1500))).toEqual([])
    })
  })

  describe('Show earlier when a refresh lands during the click', () => {
    it('still adds older rows', async () => {
      pagingRefresh()
      await pagingShowEarlier()
      await pagingShowEarlier()
      const oldestBefore = pagingStore[0].rowId!
      pagingRefresh()
      let reads = 0
      // SAFETY: the fake page returns the messages and pagination fields the backfill reads.
      vi.mocked(getOlderSessionMessages).mockImplementation(async (_id, _scope, offset) => {
        reads += 1

        if (reads === 2) {
          pagingRefresh()
        }

        // SAFETY: the page carries the messages and pagination fields the backfill reads.
        return pagingLatestPage(offset) as never
      })

      await pagingShowEarlier()

      expect(pagingStore[0].rowId).toBeLessThan(oldestBefore)
    })
  })

  describe('Show earlier when a held page opens on a newer row id', () => {
    it('still walks past it after a refresh', async () => {
      // The backend's display read keeps the newest copy of a row in its slot, so
      // a page can open on a row id newer than the next page's first row.
      pagingBackend[240] = 9001
      pagingRefresh()
      await pagingShowEarlier()
      await pagingShowEarlier()

      for (let click = 0; click < 8 && transcriptBackfillAvailable(SID); click++) {
        pagingRefresh()
        await pagingShowEarlier()
      }

      expect(pagingMissing(pagingBackend)).toEqual([])
    })
  })

  describe('Show earlier through tool turns', () => {
    it('reads past held pages of tool turns after a refresh', async () => {
      const turnRows = (turn: number): SessionMessage[] => {
        const base = turn * 4 + 1

        return [
          { id: base, role: 'user', content: `t${turn}`, timestamp: 1_000 + base },
          {
            id: base + 1,
            role: 'assistant',
            content: '',
            tool_calls: [{ id: `c${turn}`, type: 'function', function: { name: 'read_file', arguments: '{}' } }],
            timestamp: 1_001 + base
          },
          { id: base + 2, role: 'tool', tool_call_id: `c${turn}`, content: 'ok', timestamp: 1_002 + base },
          { id: base + 3, role: 'assistant', content: `reply ${turn}`, timestamp: 1_003 + base }
        ]
      }

      const rows = Array.from({ length: 150 }, (_, turn) => turnRows(turn)).flat()
      const pageAt = (offset: number) => {
        const end = rows.length - offset
        const messages = rows.slice(Math.max(0, end - LIMIT), Math.max(0, end))

        return { messages, pagination: { limit: LIMIT, offset, order: 'latest' as const, returned: messages.length } }
      }

      // SAFETY: pageAt returns the messages and pagination fields the backfill reads.
      vi.mocked(getOlderSessionMessages).mockImplementation(async (_id, _scope, offset) => pageAt(offset) as never)
      pagingStore = toChatMessages(pageAt(0).messages)
      recordTranscriptTail(SID, pageAt(0))
      await pagingShowEarlier()
      await pagingShowEarlier()
      const oldestBefore = pagingStore[0].rowId!

      recordTranscriptTail(SID, pageAt(0))
      await pagingShowEarlier()

      expect(pagingStore[0].rowId).toBeLessThan(oldestBefore)
    })
  })

  describe('Show earlier bounds one click', () => {
    const fetches = () => vi.mocked(getOlderSessionMessages).mock.calls.length

    it('stops on a backend that alternates two held pages', async () => {
      pagingRefresh()
      await pagingShowEarlier()
      await pagingShowEarlier()
      const pages = [pagingLatestPage(0), pagingLatestPage(120)]
      vi.mocked(getOlderSessionMessages).mockClear()
      // SAFETY: the fake pages carry the messages and pagination fields the backfill reads.
      vi.mocked(getOlderSessionMessages).mockImplementation(
        async (_id, _scope, offset) =>
          ({ ...pages[fetches() % 2], pagination: { ...pages[0].pagination, offset } }) as never
      )

      await pagingShowEarlier()

      expect(fetches()).toBeLessThanOrEqual(8)
    })

    it('stops on a backend that ignores the offset', async () => {
      pagingRefresh()
      vi.mocked(getOlderSessionMessages).mockClear()
      // SAFETY: the newest page with the requested offset echoed, the fields the backfill reads.
      vi.mocked(getOlderSessionMessages).mockImplementation(async (_id, _scope, offset) => {
        const page = pagingLatestPage(0)

        // SAFETY: the page carries the messages and pagination fields the backfill reads.
        return { ...page, pagination: { ...page.pagination, offset } } as never
      })

      await pagingShowEarlier()

      expect(fetches()).toBeLessThanOrEqual(8)
    })

    it('stops on a backend that renumbers rows that never render', async () => {
      pagingRefresh()
      vi.mocked(getOlderSessionMessages).mockClear()
      // SAFETY: the fake pages carry the messages and pagination fields the backfill reads.
      vi.mocked(getOlderSessionMessages).mockImplementation(
        async (_id, _scope, offset) =>
          ({
            messages: Array.from({ length: LIMIT }, (_, i) => ({
              id: 10_000 + fetches() * LIMIT + i,
              role: 'assistant',
              content: '',
              display_kind: 'hidden'
            })),
            pagination: { limit: LIMIT, offset, order: 'latest', returned: LIMIT }
          }) as never
      )

      await pagingShowEarlier()

      expect(fetches()).toBe(8)
    })

    it('keeps one fetch per click for a caller that passes no held messages', async () => {
      pagingRefresh()
      vi.mocked(getOlderSessionMessages).mockClear()
      const applied: ChatMessage[][] = []
      // SAFETY: pagingLatestPage returns the messages and pagination fields the backfill reads.
      vi.mocked(getOlderSessionMessages).mockImplementation(async (_id, _scope, offset) => {
        if (fetches() === 1) {
          rewindTranscriptTail(SID, 10)
        }

        // SAFETY: the page carries the messages and pagination fields the backfill reads.
        return pagingLatestPage(offset) as never
      })

      await backfillOlderTranscriptPage({
        storedSessionId: SID,
        isCurrent: () => true,
        applyOlderPage: older => {
          applied.push(older)
        }
      })

      expect({ fetches: fetches(), applied: applied.length }).toEqual({ fetches: 1, applied: 0 })
    })

    it('reads one page when the store holds nothing', async () => {
      recordTranscriptTail(SID, pagingLatestPage(0))
      vi.mocked(getOlderSessionMessages).mockClear()

      await pagingShowEarlier()

      expect(fetches()).toBe(1)
      expect(transcriptBackfillAvailable(SID)).toBe(true)
    })
  })
})
