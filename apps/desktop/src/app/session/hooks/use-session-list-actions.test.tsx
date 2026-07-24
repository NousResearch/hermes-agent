import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo, SidebarSessionsResponse } from '@/hermes'
import {
  $cronSessions,
  $messagingPlatformTotals,
  $messagingSessions,
  $messagingTruncated,
  $sessions,
  $sessionsLoading,
  setCronSessions,
  setMessagingPlatformTotals,
  setMessagingSessions,
  setMessagingTruncated,
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

const deferred = <T,>() => {
  let resolve!: (value: T) => void

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

// The refresh only reads the optimistic tombstone set; stub it so we don't pull
// the whole projects store (gateway / fs / git) into this hook's test.
const removed = vi.hoisted(() => ({ ids: new Set<string>() }))

vi.mock('@/store/projects', () => ({
  $removedSessionIds: { get: () => removed.ids }
}))

beforeEach(() => {
  listSidebarSessions.mockReset()
  listAllProfileSessions.mockReset()
  removed.ids = new Set()
  setSessions([])
  setCronSessions([])
  setMessagingSessions([])
  setMessagingPlatformTotals({})
  setMessagingTruncated(false)
  setSessionsLoading(false)
})

afterEach(() => {
  setSessions([])
  setCronSessions([])
  setMessagingSessions([])
  setMessagingPlatformTotals({})
  setMessagingTruncated(false)
  setSessionsLoading(false)
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

  it('drops rows the user just deleted, even when the backend page still lists them', async () => {
    // A delete RPC is in flight: the row is tombstoned optimistically but the
    // batched refresh still carries it (and a lineage-tip variant). Both must be
    // filtered so the optimistic removal never flashes back.
    removed.ids = new Set(['b', 'root-c'])
    listSidebarSessions.mockResolvedValue(
      sidebar({
        sessions: [row('a'), row('b'), row('c', { _lineage_root_id: 'root-c' } as Partial<SessionInfo>)],
        total: 3,
        profile_totals: {}
      })
    )

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    expect($sessions.get().map(s => s.id)).toEqual(['a'])
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

  it.each(['default', 'alma', 'aegis_h-01', 'synapse_h-01'])(
    'keeps every sidebar slice inside concrete profile %s',
    async profileScope => {
      const own = row(`${profileScope}-local`, { profile: profileScope })
      const ownCron = row(`${profileScope}-cron`, { profile: profileScope, source: 'cron' })
      const ownTelegram = row(`${profileScope}-telegram`, { profile: profileScope, source: 'telegram' })
      const sibling = row('sibling-local', { profile: 'sibling' })
      const siblingCron = row('sibling-cron', { profile: 'sibling', source: 'cron' })
      const siblingTelegram = row('sibling-telegram', { profile: 'sibling', source: 'telegram' })

      listSidebarSessions.mockResolvedValue(
        sidebar(
          { sessions: [own, sibling], total: 2, profile_totals: { [profileScope]: 1, sibling: 1 } },
          [ownCron, siblingCron],
          [ownTelegram, siblingTelegram]
        )
      )

      const { result } = renderHook(() => useSessionListActions({ profileScope }))

      await act(async () => {
        await result.current.refreshSessions()
      })

      expect($sessions.get().map(s => s.id)).toEqual([`${profileScope}-local`])
      expect($cronSessions.get().map(s => s.id)).toEqual([`${profileScope}-cron`])
      expect($messagingSessions.get().map(s => s.id)).toEqual([`${profileScope}-telegram`])
    }
  )

  it('keeps provenance while All Profiles aggregates every profile', async () => {
    const messaging = [
      row('default-telegram', { profile: 'default', source: 'telegram' }),
      row('alma-telegram', { profile: 'alma', source: 'telegram' })
    ]

    listSidebarSessions.mockResolvedValue(
      sidebar({ sessions: [], total: 0, profile_totals: {} }, [], messaging)
    )

    const { result } = renderHook(() => useSessionListActions({ profileScope: '__all__' }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    expect($messagingSessions.get().map(s => [s.id, s.profile])).toEqual([
      ['default-telegram', 'default'],
      ['alma-telegram', 'alma']
    ])
  })

  it('replaces default rows with Alma rows after a profile switch', async () => {
    listSidebarSessions
      .mockResolvedValueOnce(
        sidebar(
          { sessions: [row('default-local')], total: 1, profile_totals: { default: 1 } },
          [],
          [row('default-telegram', { source: 'telegram' })]
        )
      )
      .mockResolvedValueOnce(
        sidebar(
          { sessions: [row('alma-local', { profile: 'alma' })], total: 1, profile_totals: { alma: 1 } },
          [],
          [row('alma-telegram', { profile: 'alma', source: 'telegram' })]
        )
      )

    const { result, rerender } = renderHook(
      ({ scope }: { scope: string }) => useSessionListActions({ profileScope: scope }),
      { initialProps: { scope: 'default' } }
    )

    await act(async () => {
      await result.current.refreshSessions()
    })
    rerender({ scope: 'alma' })
    await act(async () => {
      await result.current.refreshSessions()
    })

    expect($sessions.get().map(s => s.id)).toEqual(['alma-local'])
    expect($messagingSessions.get().map(s => s.id)).toEqual(['alma-telegram'])
  })

  it('clears Messaging pagination state when the profile scope changes', () => {
    setMessagingPlatformTotals({ telegram: 27 })
    setMessagingTruncated(true)

    const { rerender } = renderHook(
      ({ scope }: { scope: string }) => useSessionListActions({ profileScope: scope }),
      { initialProps: { scope: 'default' } }
    )

    expect($messagingPlatformTotals.get()).toEqual({})
    expect($messagingTruncated.get()).toBe(false)

    setMessagingPlatformTotals({ telegram: 11 })
    setMessagingTruncated(true)
    rerender({ scope: 'alma' })

    expect($messagingPlatformTotals.get()).toEqual({})
    expect($messagingTruncated.get()).toBe(false)
  })

  it('does not let a delayed previous-profile refresh overwrite the new scope', async () => {
    const oldRequest = deferred<SidebarSessionsResponse>()
    listSidebarSessions
      .mockReturnValueOnce(oldRequest.promise)
      .mockResolvedValueOnce(
        sidebar(
          { sessions: [row('alma-local', { profile: 'alma' })], total: 1, profile_totals: { alma: 1 } },
          [],
          [row('alma-telegram', { profile: 'alma', source: 'telegram' })]
        )
      )

    const { result, rerender } = renderHook(
      ({ scope }: { scope: string }) => useSessionListActions({ profileScope: scope }),
      { initialProps: { scope: 'default' } }
    )

    const delayed = result.current.refreshSessions()

    rerender({ scope: 'alma' })
    await act(async () => {
      await result.current.refreshSessions()
    })

    oldRequest.resolve(
      sidebar(
        { sessions: [row('default-local')], total: 1, profile_totals: { default: 1 } },
        [],
        [row('default-telegram', { source: 'telegram' })]
      )
    )
    await act(async () => {
      await delayed
    })

    expect($sessions.get().map(s => s.id)).toEqual(['alma-local'])
    expect($messagingSessions.get().map(s => s.id)).toEqual(['alma-telegram'])
  })

  it('preserves the concrete profile when loading more Messaging rows', async () => {
    listAllProfileSessions.mockResolvedValue({
      sessions: [row('alma-telegram', { profile: 'alma', source: 'telegram' })],
      total: 1
    })
    const { result } = renderHook(() => useSessionListActions({ profileScope: 'alma' }))

    await act(async () => {
      await result.current.loadMoreMessagingForPlatform('telegram')
    })

    expect(listAllProfileSessions).toHaveBeenCalledWith(
      expect.any(Number),
      1,
      'exclude',
      'recent',
      'alma',
      { source: 'telegram' }
    )
    expect($messagingSessions.get().map(s => s.id)).toEqual(['alma-telegram'])
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
