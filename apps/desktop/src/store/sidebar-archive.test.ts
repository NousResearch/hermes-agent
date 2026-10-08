import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { $archivedSessions, listEveryArchivedSession, loadArchivedSessions } from './sidebar-archive'

const listAllProfileSessions = vi.hoisted(() => vi.fn())

vi.mock('@/api/sessions', () => ({
  listAllProfileSessions
}))

const PAGE_SIZE = 200

const row = (id: string, overrides: Partial<SessionInfo> = {}): SessionInfo =>
  ({
    id,
    archived: true,
    input_tokens: 0,
    output_tokens: 0,
    message_count: 1,
    started_at: 1,
    last_active: 1,
    source: 'desktop',
    tool_call_count: 0,
    ...overrides
  }) as SessionInfo

/** A back-filled pinned row: appended past the window by the server, ignores
 * the archived filter, and repeats on every page whose window covers it. */
const pinnedRow = (id: string): SessionInfo => row(id, { archived: false, pinned: true })

/** `id`-numbered archived rows for a full page of the given size. */
const page = (from: number, count: number): SessionInfo[] =>
  Array.from({ length: count }, (_, i) => row(`a-${from + i}`))

beforeEach(() => {
  $archivedSessions.set([])
  listAllProfileSessions.mockReset()
})

describe('loadArchivedSessions', () => {
  it('keeps the last successful result when a refresh fails', async () => {
    const existing = { id: 'archived-1', title: 'Keep me' } as SessionInfo
    $archivedSessions.set([existing])
    listAllProfileSessions.mockRejectedValue(new Error('offline'))

    await loadArchivedSessions()

    expect($archivedSessions.get()).toEqual([existing])
  })
})

describe('listEveryArchivedSession', () => {
  it('paginates until the backend total is loaded', async () => {
    const firstPage = Array.from({ length: 200 }, (_, i) => row(`a${i}`))

    listAllProfileSessions
      .mockResolvedValueOnce({ sessions: firstPage, total: 201, limit: 200, offset: 0 })
      .mockResolvedValueOnce({ sessions: [row('c')], total: 201, limit: 200, offset: 200 })

    await expect(listEveryArchivedSession()).resolves.toEqual([...firstPage, row('c')])
    expect(listAllProfileSessions).toHaveBeenNthCalledWith(1, 200, 0, 'only', 'recent', 'all', {}, 0)
    expect(listAllProfileSessions).toHaveBeenNthCalledWith(2, 200, 0, 'only', 'recent', 'all', {}, 200)
  })

  it('stops on an empty page when profiles changed during pagination', async () => {
    const firstPage = Array.from({ length: 200 }, (_, i) => row(`a${i}`))

    listAllProfileSessions
      .mockResolvedValueOnce({ sessions: firstPage, total: 400, limit: 200, offset: 0 })
      .mockResolvedValueOnce({ sessions: [], total: 400, limit: 200, offset: 200 })

    await expect(listEveryArchivedSession()).resolves.toEqual(firstPage)
    expect(listAllProfileSessions).toHaveBeenCalledTimes(2)
  })

  it('advances the offset by the page size, not by the row count', async () => {
    // Server contract: a page returns limit window rows PLUS every back-filled
    // pinned row, so pages 1 and 2 here both hand back 200 archived rows and
    // a riding pin. The next offset must stay 200/400 regardless.
    listAllProfileSessions
      .mockResolvedValueOnce({
        sessions: [...page(0, PAGE_SIZE), pinnedRow('pin-x')],
        total: 401,
        limit: PAGE_SIZE,
        offset: 0
      })
      .mockResolvedValueOnce({
        sessions: [...page(PAGE_SIZE, PAGE_SIZE), pinnedRow('pin-x')],
        total: 401,
        limit: PAGE_SIZE,
        offset: PAGE_SIZE
      })
      .mockResolvedValueOnce({ sessions: [row('a-400')], total: 401, limit: PAGE_SIZE, offset: 2 * PAGE_SIZE })

    const sessions = await listEveryArchivedSession()

    // All 401 archived rows plus the pin — none skipped, pin listed once.
    expect(sessions).toHaveLength(402)
    expect(sessions.filter(s => s.id === 'pin-x')).toHaveLength(1)
    expect(sessions.filter(s => s.archived).map(s => s.id)).toContain('a-400')
    expect(listAllProfileSessions).toHaveBeenNthCalledWith(2, PAGE_SIZE, 0, 'only', 'recent', 'all', {}, PAGE_SIZE)
    expect(listAllProfileSessions).toHaveBeenNthCalledWith(3, PAGE_SIZE, 0, 'only', 'recent', 'all', {}, 2 * PAGE_SIZE)
  })

  it('loads every archived row when non-archived pins inflate each page', async () => {
    // Reviewer's worst case: 1000 archived rows; pins live outside the archive
    // and total counts only archived rows. pages 1-5 are full 200-row windows
    // each carrying 100 extra pin rows, so the row count crosses 1000 on page
    // 5 — the loop must still stop on the cursor, not on the inflated count.
    const total = 1000
    let calls = 0
    listAllProfileSessions.mockImplementation(async (...args: unknown[]) => {
      const offset = args[6] as number
      calls += 1
      const remaining = Math.max(0, Math.min(PAGE_SIZE, total - offset))
      const pins = Array.from({ length: 100 }, (_, i) => pinnedRow(`pin-${i}`))

      return {
        sessions: [...page(offset, remaining), ...pins],
        total,
        limit: PAGE_SIZE,
        offset
      }
    })

    const sessions = await listEveryArchivedSession()

    // 5 pages: offset 0,200,400,600,800 — the 6th request is never issued
    // because offset 1000 has reached total.
    expect(calls).toBe(5)
    expect(sessions.filter(s => s.archived)).toHaveLength(total)
    // The 100 distinct pins ride along once each, not once per page.
    expect(sessions.filter(s => !s.archived)).toHaveLength(100)
  })
})
