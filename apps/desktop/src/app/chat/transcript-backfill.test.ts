import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { $transcriptTailBySessionId, recordTranscriptTail, transcriptTailState } from '@/store/transcript-tail'

import {
  _resetTranscriptBackfillForTests,
  backfillOlderTranscriptPage,
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

  it('returns the refreshed tail when it is not shorter than the previous transcript', () => {
    const previous = [chat('a', 1)]
    const refreshed = [chat('a', 1), chat('b', 2)]

    // Both rows already carry their durable id, so there is no identity to carry
    // and the tail is adopted untouched — same array, same objects.
    expect(graftRefreshedTailOntoBackfill(refreshed, previous)).toBe(refreshed)
  })

  it('carries a live row’s render identity onto its refreshed twin', () => {
    const previous: ChatMessage[] = [
      { id: 'user-1700000000000-abc123', role: 'user', parts: [{ type: 'text', text: 'hello' }], pending: true },
      {
        id: 'assistant-stream-live-1',
        role: 'assistant',
        parts: [{ type: 'reasoning', text: 'thinking about it' }],
        pending: true
      }
    ]

    // What the turn-end refresh hands back: the same turn, committed ids.
    const refreshed: ChatMessage[] = [
      { id: '1789333950.13104-0-user', role: 'user', parts: [{ type: 'text', text: 'hello' }], rowId: 42 },
      {
        id: '1789333950.13104-1-assistant',
        role: 'assistant',
        parts: [
          { type: 'reasoning', text: 'thinking about it' },
          { type: 'text', text: 'the answer' }
        ],
        rowId: 43
      }
    ]

    const grafted = graftRefreshedTailOntoBackfill(refreshed, previous)

    // The committed rows win on id, but the identity the live rows were rendering
    // under rides along — including the reasoning-only row, which pairs on its
    // reasoning text because it has no answer text yet.
    expect(grafted.map(message => message.id)).toEqual(['1789333950.13104-0-user', '1789333950.13104-1-assistant'])
    expect(grafted.map(message => message.rowKey)).toEqual(['user-1700000000000-abc123', 'assistant-stream-live-1'])
  })

  it('carries a tool-only row’s identity, which has no text to pair on', () => {
    const toolPart = [{ type: 'tool-call', toolName: 'read_file', toolCallId: 'call-1' }] as ChatMessage['parts']

    const previous: ChatMessage[] = [
      { id: 'assistant-stream-tool-1', role: 'assistant', parts: toolPart, pending: true }
    ]

    const refreshed: ChatMessage[] = [
      { id: '1789333950.13104-1-assistant', role: 'assistant', parts: toolPart, rowId: 51 }
    ]

    // A tool-only row is a live row like any other: its tool disclosure must survive
    // the refresh, so it pairs on its tool signature.
    expect(graftRefreshedTailOntoBackfill(refreshed, previous)[0].rowKey).toBe('assistant-stream-tool-1')
  })

  it('pairs on the durable row id even when the reply was rewritten', () => {
    const previous: ChatMessage[] = [
      {
        id: 'assistant-stream-9',
        role: 'assistant',
        parts: [{ type: 'text', text: 'an interim reply' }],
        rowId: 77,
        pending: true
      }
    ]

    const refreshed: ChatMessage[] = [
      {
        id: '1789333950.13104-1-assistant',
        role: 'assistant',
        parts: [{ type: 'text', text: 'the rewritten reply' }],
        rowId: 77
      }
    ]

    // Same row id is the same row, however far the text moved — a verify-on-stop
    // reply shares no prefix with the interim it replaces.
    expect(graftRefreshedTailOntoBackfill(refreshed, previous)[0].rowKey).toBe('assistant-stream-9')
  })

  it('does not let a short live prompt claim a longer older row', () => {
    const previous: ChatMessage[] = [
      { id: 'user-old', role: 'user', parts: [{ type: 'text', text: 'help me write an essay about rivers' }] }
    ]

    const refreshed: ChatMessage[] = [
      { id: '1789333950.13104-0-user', role: 'user', parts: [{ type: 'text', text: 'help' }], rowId: 40 }
    ]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous)[0].rowKey).toBeUndefined()
  })

  it('never hands a refreshed row an identity the graft keeps in the array', () => {
    const previous: ChatMessage[] = [
      // Unpersisted and about to be KEPT by the graft: nothing may inherit its identity.
      { id: 'user-unpersisted-old', role: 'user', parts: [{ type: 'text', text: 'hello' }] },
      { id: 'prev-committed', role: 'user', parts: [{ type: 'text', text: 'other' }], rowId: 2 }
    ]

    const refreshed: ChatMessage[] = [
      { id: '1789333950.13104-1-user', role: 'user', parts: [{ type: 'text', text: 'other' }], rowId: 2 },
      { id: '1789333950.13104-2-user', role: 'user', parts: [{ type: 'text', text: 'hello' }], rowId: 3 }
    ]

    const grafted = graftRefreshedTailOntoBackfill(refreshed, previous)
    const identities = grafted.map(message => message.rowKey ?? message.id)

    // Two rows sharing one identity is worse than a remount: it is a duplicate key.
    expect(new Set(identities).size).toBe(identities.length)

    // The replaced row keeps the identity it was rendering under — same durable row
    // id, so the refresh must not re-key it...
    expect(identities[1]).toBe('prev-committed')

    // ...while 'hello' must NOT inherit the identity of the row the graft KEEPS.
    expect(grafted[2].rowKey).toBeUndefined()
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

  it('shares one in-flight fetch per stored session', async () => {
    truncatedTail()

    let resolvePage: (value: unknown) => void = () => {}

    vi.mocked(getOlderSessionMessages).mockReturnValue(
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
