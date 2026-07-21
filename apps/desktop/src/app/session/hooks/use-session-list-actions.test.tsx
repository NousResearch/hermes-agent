import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo, SidebarSessionsResponse } from '@/hermes'
import { $removedSessionIds, tombstoneSessions } from '@/store/projects'
import {
  $cronSessions,
  $messagingPlatformTotals,
  $messagingSessions,
  $sessions,
  $sessionsLoading,
  setCronSessions,
  setMessagingPlatformTotals,
  setMessagingSessions,
  setSessions,
  setSessionsLoading
} from '@/store/session'

import { useSessionListActions } from './use-session-list-actions'

// Sidebar refresh hygiene: a content-identical refresh (turn complete,
// cross-window broadcast, reconnect) must not replace $sessions' array
// identity — that identity is the dependency for every sidebar memo — and
// must not flicker the loading flag over an already-populated list.

const row = (id: string, over: Partial<SessionInfo> = {}): SessionInfo =>
  ({
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: 1000,
    message_count: 3,
    model: 'm',
    output_tokens: 0,
    preview: 'hey',
    profile: 'default',
    source: 'desktop',
    started_at: 900,
    title: `Chat ${id}`,
    ...over
  }) as SessionInfo

// Batched sidebar response builder. `refreshSessions` now makes ONE
// listSidebarSessions call that returns all three slices, replacing the three
// separate listAllProfileSessions calls (each of which reopened every profile
// DB) — #66377-adjacent perf work from the desktop audit canvas.
const sidebar = (
  recents: { sessions: SessionInfo[]; total?: number; profile_totals?: Record<string, number> },
  cron: SessionInfo[] = [],
  messaging: SessionInfo[] = []
): SidebarSessionsResponse => ({
  recents: { sessions: recents.sessions, total: recents.total, profile_totals: recents.profile_totals },
  cron: { sessions: cron },
  messaging: { sessions: messaging, total: messaging.length }
})

const listSidebarSessions = vi.fn()
const listAllProfileSessions = vi.fn()

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobs: vi.fn(async () => []),
  listAllProfileSessions: (...args: unknown[]) => listAllProfileSessions(...args),
  listSidebarSessions: (...args: unknown[]) => listSidebarSessions(...args)
}))

beforeEach(() => {
  listSidebarSessions.mockReset()
  listAllProfileSessions.mockReset()
  setSessions([])
  setCronSessions([])
  setMessagingPlatformTotals({})
  setMessagingSessions([])
  setSessionsLoading(false)
  $removedSessionIds.set(new Set())
})

afterEach(() => {
  setSessions([])
  setCronSessions([])
  setMessagingPlatformTotals({})
  setMessagingSessions([])
  setSessionsLoading(false)
  $removedSessionIds.set(new Set())
  vi.useRealTimers()
})

describe('refreshSessions identity + loading hygiene', () => {
  it('keeps the previous $sessions array when the refresh is content-identical', async () => {
    const rows = [row('a'), row('b')]
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: rows, total: 2, profile_totals: { default: 2 } }))

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    const first = $sessions.get()
    expect(first.map(s => s.id)).toEqual(['a', 'b'])

    // Second refresh returns fresh (but equal) row objects, as the API does.
    listSidebarSessions.mockResolvedValue(
      sidebar({ sessions: [row('a'), row('b')], total: 2, profile_totals: { default: 2 } })
    )

    await act(async () => {
      await result.current.refreshSessions()
    })

    expect($sessions.get()).toBe(first)
  })

  it('swaps the array when rows actually changed', async () => {
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [row('a')], total: 1, profile_totals: {} }))
    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    const first = $sessions.get()

    listSidebarSessions.mockResolvedValue(
      sidebar({ sessions: [row('a', { last_active: 2000, title: 'Renamed' })], total: 1, profile_totals: {} })
    )

    await act(async () => {
      await result.current.refreshSessions()
    })

    expect($sessions.get()).not.toBe(first)
    expect($sessions.get()[0].title).toBe('Renamed')
  })

  it('does not flicker the loading flag over a populated list', async () => {
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [row('a')], total: 1, profile_totals: {} }))
    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    const loadingStates: boolean[] = []
    const off = $sessionsLoading.subscribe(value => loadingStates.push(value))

    await act(async () => {
      await result.current.refreshSessions()
    })

    off()
    // Only the initial subscribe emission — no true/false churn per refresh.
    expect(loadingStates).toEqual([false])
  })

  it('still shows loading for the initial (empty-list) fetch', async () => {
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [row('a')], total: 1, profile_totals: {} }))
    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    const loadingStates: boolean[] = []
    const off = $sessionsLoading.subscribe(value => loadingStates.push(value))

    await act(async () => {
      await result.current.refreshSessions()
    })

    off()
    expect(loadingStates).toEqual([false, true, false])
  })
})

describe('refreshSessions batches slices into one request', () => {
  it('makes a single sidebar call and distributes recents / cron / messaging', async () => {
    const recents = [row('a'), row('b')]
    const cron = [row('c1', { source: 'cron', title: 'nightly' })]
    const messaging = [row('m1', { source: 'telegram', title: 'tg chat' })]

    listSidebarSessions.mockResolvedValue(
      sidebar({ sessions: recents, total: 2, profile_totals: { default: 2 } }, cron, messaging)
    )

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    // One batched call, not three separate listAllProfileSessions reads.
    expect(listSidebarSessions).toHaveBeenCalledTimes(1)
    expect(listAllProfileSessions).not.toHaveBeenCalled()

    // Each slice landed in its own store.
    expect($sessions.get().map(s => s.id)).toEqual(['a', 'b'])
    expect($cronSessions.get().map(s => s.id)).toEqual(['c1'])
    expect($messagingSessions.get().map(s => s.id)).toEqual(['m1'])
  })

  it('filters optimistic removals from recents, cron, and messaging', async () => {
    tombstoneSessions(['recent-gone', 'cron-gone', 'message-gone'])
    listSidebarSessions.mockResolvedValue(
      sidebar(
        { sessions: [row('recent-gone'), row('recent-kept')], total: 2, profile_totals: { default: 2 } },
        [row('cron-gone', { source: 'cron' }), row('cron-kept', { source: 'cron' })],
        [row('message-gone', { source: 'telegram' }), row('message-kept', { source: 'telegram' })]
      )
    )

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    expect($sessions.get().map(s => s.id)).toEqual(['recent-kept'])
    expect($cronSessions.get().map(s => s.id)).toEqual(['cron-kept'])
    expect($messagingSessions.get().map(s => s.id)).toEqual(['message-kept'])
  })

  it('retains the hide marker across a delayed response and post-mutation refresh', async () => {
    vi.useFakeTimers()
    const stale = deferred<{ sessions: SessionInfo[]; total: number }>()
    listAllProfileSessions.mockReturnValueOnce(stale.promise)
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [], total: 0, profile_totals: {} }))

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))
    const staleRefresh = result.current.loadMoreMessagingForPlatform('telegram')

    tombstoneSessions(['message-gone'])

    await act(async () => {
      await result.current.refreshSessions()
    })

    await vi.advanceTimersByTimeAsync(2_100)
    expect($removedSessionIds.get().has('message-gone')).toBe(true)

    stale.resolve({ sessions: [row('message-gone', { source: 'telegram' })], total: 5 })

    await act(async () => {
      await staleRefresh
    })

    expect($messagingSessions.get()).toEqual([])
    expect($messagingPlatformTotals.get()).toEqual({})
    expect($removedSessionIds.get().has('message-gone')).toBe(true)
  })

  it('filters optimistic removals from per-platform and per-profile paging', async () => {
    tombstoneSessions(['message-gone', 'profile-gone'])
    listAllProfileSessions
      .mockResolvedValueOnce({
        sessions: [row('message-gone', { source: 'telegram' }), row('message-kept', { source: 'telegram' })],
        total: 2
      })
      .mockResolvedValueOnce({
        sessions: [row('profile-gone', { profile: 'work' }), row('profile-kept', { profile: 'work' })],
        total: 2
      })

    const { result } = renderHook(() => useSessionListActions({ profileScope: '__all__' }))

    await act(async () => {
      await result.current.loadMoreMessagingForPlatform('telegram')
      await result.current.loadMoreSessionsForProfile('work')
    })

    expect($messagingSessions.get().map(s => s.id)).toEqual(['message-kept'])
    expect($sessions.get().map(s => s.id)).toEqual(['profile-kept'])
  })

  it('forwards the active profile scope + section limits to the batched call', async () => {
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [], total: 0, profile_totals: {} }))
    const { result } = renderHook(() => useSessionListActions({ profileScope: 'work' }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    expect(listSidebarSessions).toHaveBeenCalledWith(
      expect.objectContaining({
        recentsProfile: 'work',
        recentsExclude: expect.arrayContaining(['cron']),
        messagingExclude: expect.arrayContaining(['cron'])
      })
    )
  })

  it('scopes the cron-jobs fetch to the active profile (all → unified view)', async () => {
    const { getCronJobs } = await import('@/hermes')
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [], total: 0, profile_totals: {} }))

    const scoped = renderHook(() => useSessionListActions({ profileScope: 'work' }))

    await act(async () => {
      await scoped.result.current.refreshCronJobs()
    })

    expect(getCronJobs).toHaveBeenLastCalledWith('work')

    const unified = renderHook(() => useSessionListActions({ profileScope: '__all__' }))

    await act(async () => {
      await unified.result.current.refreshCronJobs()
    })

    expect(getCronJobs).toHaveBeenLastCalledWith('all')
  })
})
