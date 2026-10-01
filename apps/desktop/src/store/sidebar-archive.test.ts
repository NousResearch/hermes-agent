import { beforeEach, describe, expect, it, vi } from 'vitest'

import { listAllProfileSessions, type SessionInfo } from '@/hermes'

import { $archivedSessions, loadArchivedSessions } from './sidebar-archive'

vi.mock('@/hermes', () => ({
  listAllProfileSessions: vi.fn()
}))

describe('loadArchivedSessions', () => {
  beforeEach(() => {
    $archivedSessions.set([])
    vi.mocked(listAllProfileSessions).mockReset()
  })

  it('keeps the last successful result when a refresh fails', async () => {
    const existing = { id: 'archived-1', title: 'Keep me' } as SessionInfo
    $archivedSessions.set([existing])
    vi.mocked(listAllProfileSessions).mockRejectedValue(new Error('offline'))

    await loadArchivedSessions()

    expect($archivedSessions.get()).toEqual([existing])
  })
})
