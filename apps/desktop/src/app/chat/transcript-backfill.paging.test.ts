import { beforeEach, describe, expect, it, vi } from 'vitest'

import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import { $transcriptTailBySessionId, recordTranscriptTail, rewindTranscriptTail } from '@/store/transcript-tail'
import type { SessionMessage } from '@/types/hermes'

import {
  _resetTranscriptBackfillForTests,
  backfillOlderTranscriptPage,
  graftRefreshedTailOntoBackfill,
  mergeOlderTranscriptPage,
  transcriptBackfillAvailable
} from './transcript-backfill'

vi.mock('@/hermes', () => ({
  getOlderSessionMessages: vi.fn(),
  getSessionMessages: vi.fn()
}))

const { getOlderSessionMessages, getSessionMessages } = await import('@/hermes')

// A stored SessionMessage row: distinct timestamps keep toChatMessages ids
// unique and the row id survives as ChatMessage.rowId.
const row = (rowId: number, text: string) => ({
  id: rowId,
  role: 'user' as const,
  content: text,
  timestamp: 1_000 + rowId
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
