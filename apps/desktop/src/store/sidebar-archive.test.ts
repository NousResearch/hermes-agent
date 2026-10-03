import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { $archivedSessions, listEveryArchivedSession, loadArchivedSessions } from './sidebar-archive'

const listAllProfileSessions = vi.hoisted(() => vi.fn())

vi.mock('@/api/sessions', () => ({
  listAllProfileSessions
}))

const row = (id: string): SessionInfo =>
  ({
    id,
    archived: true,
    input_tokens: 0,
    output_tokens: 0,
    message_count: 1,
    started_at: 1,
    last_active: 1,
    source: 'desktop',
    tool_call_count: 0
  }) as SessionInfo

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
    listAllProfileSessions
      .mockResolvedValueOnce({ sessions: [row('a'), row('b')], total: 3, limit: 200, offset: 0 })
      .mockResolvedValueOnce({ sessions: [row('c')], total: 3, limit: 200, offset: 2 })

    await expect(listEveryArchivedSession()).resolves.toEqual([row('a'), row('b'), row('c')])
    expect(listAllProfileSessions).toHaveBeenNthCalledWith(1, 200, 0, 'only', 'recent', 'all', {}, 0)
    expect(listAllProfileSessions).toHaveBeenNthCalledWith(2, 200, 0, 'only', 'recent', 'all', {}, 2)
  })

  it('stops on an empty page when profiles changed during pagination', async () => {
    listAllProfileSessions
      .mockResolvedValueOnce({ sessions: [row('a')], total: 2, limit: 200, offset: 0 })
      .mockResolvedValueOnce({ sessions: [], total: 2, limit: 200, offset: 1 })

    await expect(listEveryArchivedSession()).resolves.toEqual([row('a')])
    expect(listAllProfileSessions).toHaveBeenCalledTimes(2)
  })
})
