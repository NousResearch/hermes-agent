// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getCronJobs, listAllProfileSessions, type PaginatedSessions, type SessionInfo } from '@/hermes'
import { ALL_PROFILES } from '@/store/profile'
import {
  $messagingPlatformTotals,
  $messagingSessions,
  setMessagingPlatformTotals,
  setMessagingSessions
} from '@/store/session'

import { useSessionListActions } from './use-session-list-actions'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobs: vi.fn(),
  listAllProfileSessions: vi.fn()
}))

const emptyPage: PaginatedSessions = {
  limit: 50,
  offset: 0,
  profile_totals: {},
  sessions: [],
  total: 0
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

function messagingSession(id: string, profile: string): SessionInfo {
  return {
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    profile,
    source: 'telegram',
    started_at: 1,
    title: id,
    tool_call_count: 0
  }
}

describe('useSessionListActions messaging scope', () => {
  beforeEach(() => {
    setMessagingSessions([])
    setMessagingPlatformTotals({})
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

  it('ignores an older profile response that resolves after the active profile', async () => {
    const first = deferred<typeof emptyPage>()
    const second = deferred<typeof emptyPage>()

    vi.mocked(listAllProfileSessions)
      .mockReset()
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise)

    const { rerender, result } = renderHook(({ profileScope }) => useSessionListActions({ profileScope }), {
      initialProps: { profileScope: 'nolan' }
    })

    let firstRequest!: Promise<void>
    act(() => {
      firstRequest = result.current.refreshMessagingSessions()
    })

    rerender({ profileScope: 'silas' })

    let secondRequest!: Promise<void>
    act(() => {
      secondRequest = result.current.refreshMessagingSessions()
    })

    await act(async () => {
      second.resolve({ ...emptyPage, sessions: [messagingSession('silas-session', 'silas')], total: 1 })
      await secondRequest
    })

    await act(async () => {
      first.resolve({ ...emptyPage, sessions: [messagingSession('nolan-session', 'nolan')], total: 1 })
      await firstRequest
    })

    expect($messagingSessions.get().map(session => session.id)).toEqual(['silas-session'])
  })

  it('ignores a load-more response after the active profile changes', async () => {
    const page = deferred<typeof emptyPage>()

    vi.mocked(listAllProfileSessions)
      .mockReset()
      .mockImplementationOnce(() => page.promise)
    setMessagingSessions([messagingSession('nolan-seed', 'nolan')])
    setMessagingPlatformTotals({ telegram: 1 })

    const { rerender, result } = renderHook(({ profileScope }) => useSessionListActions({ profileScope }), {
      initialProps: { profileScope: 'nolan' }
    })

    let loadMoreRequest!: Promise<void>
    act(() => {
      loadMoreRequest = result.current.loadMoreMessagingForPlatform('telegram')
    })

    rerender({ profileScope: 'silas' })
    setMessagingSessions([messagingSession('silas-session', 'silas')])

    await act(async () => {
      page.resolve({
        ...emptyPage,
        sessions: [messagingSession('nolan-more', 'nolan')],
        total: 2
      })
      await loadMoreRequest
    })

    expect($messagingSessions.get().map(session => session.id)).toEqual(['silas-session'])
    expect($messagingPlatformTotals.get()).toEqual({ telegram: 1 })
  })
})
