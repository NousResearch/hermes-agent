import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { listAllProfileSessions } from '@/hermes'
import { $messagingSessions, setMessagingSessions } from '@/store/session'

import { useSessionListActions } from './use-session-list-actions'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobs: vi.fn(async () => []),
  listAllProfileSessions: vi.fn(async () => ({
    limit: 0,
    offset: 0,
    sessions: [],
    total: 0
  }))
}))

const listAllProfileSessionsMock = vi.mocked(listAllProfileSessions)

describe('useSessionListActions messaging profile scoping', () => {
  afterEach(() => {
    cleanup()
    setMessagingSessions([])
    vi.clearAllMocks()
  })

  it('fetches messaging sessions from the active profile instead of all profiles', async () => {
    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    await act(async () => {
      await result.current.refreshMessagingSessions()
    })

    expect(listAllProfileSessionsMock).toHaveBeenCalledWith(
      expect.any(Number),
      1,
      'exclude',
      'recent',
      'default',
      expect.objectContaining({
        excludeSources: expect.arrayContaining(['cron', 'cli', 'tui', 'desktop'])
      })
    )
  })

  it('keeps the cross-profile messaging slice only in the explicit all-profiles view', async () => {
    const { result } = renderHook(() => useSessionListActions({ profileScope: '__all__' }))

    await act(async () => {
      await result.current.refreshMessagingSessions()
    })

    expect(listAllProfileSessionsMock).toHaveBeenCalledWith(
      expect.any(Number),
      1,
      'exclude',
      'recent',
      'all',
      expect.any(Object)
    )
  })

  it('uses the active profile when loading more rows for a single messaging platform', async () => {
    $messagingSessions.set([
      {
        ended_at: null,
        id: 'telegram-1',
        input_tokens: 0,
        is_active: false,
        last_active: 1,
        message_count: 1,
        model: null,
        output_tokens: 0,
        preview: null,
        source: 'telegram',
        started_at: 1,
        title: 'telegram chat',
        tool_call_count: 0
      }
    ])

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'telegram' }))

    await act(async () => {
      await result.current.loadMoreMessagingForPlatform('telegram')
    })

    expect(listAllProfileSessionsMock).toHaveBeenCalledWith(
      expect.any(Number),
      1,
      'exclude',
      'recent',
      'telegram',
      { source: 'telegram' }
    )
  })
})
