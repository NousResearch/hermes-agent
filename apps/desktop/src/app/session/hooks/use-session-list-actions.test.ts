// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getCronJobs, listAllProfileSessions } from '@/hermes'
import { ALL_PROFILES } from '@/store/profile'
import { setMessagingSessions } from '@/store/session'

import { useSessionListActions } from './use-session-list-actions'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobs: vi.fn(),
  listAllProfileSessions: vi.fn()
}))

const emptyPage = {
  limit: 50,
  offset: 0,
  profile_totals: {},
  sessions: [],
  total: 0
}

describe('useSessionListActions messaging scope', () => {
  beforeEach(() => {
    setMessagingSessions([])
    vi.mocked(getCronJobs).mockResolvedValue([])
    vi.mocked(listAllProfileSessions).mockResolvedValue(emptyPage)
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('fetches messaging sessions only for the active profile', async () => {
    const { result } = renderHook(() => useSessionListActions({ profileScope: 'nolan' }))

    await act(async () => {
      await result.current.refreshMessagingSessions()
    })

    expect(listAllProfileSessions).toHaveBeenCalledWith(
      expect.any(Number),
      1,
      'exclude',
      'recent',
      'nolan',
      expect.objectContaining({ excludeSources: expect.any(Array) })
    )
  })

  it('keeps messaging global in the explicit all-profiles view', async () => {
    const { result } = renderHook(() => useSessionListActions({ profileScope: ALL_PROFILES }))

    await act(async () => {
      await result.current.refreshMessagingSessions()
    })

    expect(listAllProfileSessions).toHaveBeenCalledWith(
      expect.any(Number),
      1,
      'exclude',
      'recent',
      'all',
      expect.any(Object)
    )
  })

  it('pages one messaging platform within the active profile', async () => {
    const { result } = renderHook(() => useSessionListActions({ profileScope: 'silas' }))

    await act(async () => {
      await result.current.loadMoreMessagingForPlatform('slack')
    })

    expect(listAllProfileSessions).toHaveBeenCalledWith(expect.any(Number), 1, 'exclude', 'recent', 'silas', {
      source: 'slack'
    })
  })
})
