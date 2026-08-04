import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo, SidebarSessionsResponse } from '@/hermes'
import { $pinnedSessionIds } from '@/store/layout'
import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  $sessionsLoading,
  setCronSessions,
  setMessagingSessions,
  setSessions,
  setSessionsLoading,
  setSessionsTotal
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
  recents: { sessions: SessionInfo[]; profiles_truncated?: Record<string, boolean> },
  cron: SessionInfo[] = [],
  messaging: SessionInfo[] = []
): SidebarSessionsResponse => ({
  recents: { sessions: recents.sessions, profiles_truncated: recents.profiles_truncated },
  cron: { sessions: cron },
  messaging: { sessions: messaging }
})

const listSidebarSessions = vi.fn()
const listAllProfileSessions = vi.fn()
const bulkDeleteSessions = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  bulkDeleteSessions: (...args: unknown[]) => bulkDeleteSessions(...args),
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
  bulkDeleteSessions.mockReset()
  removed.ids = new Set()
  setSessions([])
  setCronSessions([])
  setMessagingSessions([])
  setSessionsLoading(false)
  setSessionsTotal(0)
  $pinnedSessionIds.set([])
})

afterEach(() => {
  setSessions([])
  setCronSessions([])
  setMessagingSessions([])
  setSessionsLoading(false)
  setSessionsTotal(0)
  $pinnedSessionIds.set([])
})

describe('refreshSessions identity + loading hygiene', () => {
  it('keeps the previous $sessions array when the refresh is content-identical', async () => {
    const rows = [row('a'), row('b')]
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: rows }))

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    const first = $sessions.get()
    expect(first.map(s => s.id)).toEqual(['a', 'b'])

    // Second refresh returns fresh (but equal) row objects, as the API does.
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [row('a'), row('b')] }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    expect($sessions.get()).toBe(first)
  })

  it('swaps the array when rows actually changed', async () => {
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [row('a')] }))
    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    const first = $sessions.get()

    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [row('a', { last_active: 2000, title: 'Renamed' })] }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    expect($sessions.get()).not.toBe(first)
    expect($sessions.get()[0].title).toBe('Renamed')
  })

  it('does not flicker the loading flag over a populated list', async () => {
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [row('a')] }))
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
        sessions: [row('a'), row('b'), row('c', { _lineage_root_id: 'root-c' } as Partial<SessionInfo>)]
      })
    )

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    expect($sessions.get().map(s => s.id)).toEqual(['a'])
  })

  it('still shows loading for the initial (empty-list) fetch', async () => {
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [row('a')] }))
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

    listSidebarSessions.mockResolvedValue(sidebar({ sessions: recents }, cron, messaging))

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
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [] }))
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
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [] }))

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

describe('clearAllSessions', () => {
  // The clear loop pages the recents scope in BULK_DELETE_MAX_IDS-row chunks
  // via listAllProfileSessions; the closing (finally) refresh goes through the
  // batched sidebar endpoint. Keep the latter empty unless a test needs
  // survivors to reconcile against.
  const emptyPage = { limit: 0, offset: 0, sessions: [] as SessionInfo[], total: 0 }

  beforeEach(() => {
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [], total: 0, profile_totals: {} }))
  })

  it('pages the scope, bulk-deletes every chat, and clears the list + pins', async () => {
    const rows = [row('s1'), row('s2')]
    setSessions(rows)
    setSessionsTotal(2)
    $pinnedSessionIds.set(['s2'])

    listAllProfileSessions
      .mockResolvedValueOnce({ limit: 500, offset: 0, sessions: rows, total: 2 })
      .mockResolvedValue(emptyPage)
    bulkDeleteSessions.mockImplementation((ids: string[]) => Promise.resolve({ deleted: ids.length, ok: true }))

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    let removedCount = 0

    await act(async () => {
      removedCount = await result.current.clearAllSessions()
    })

    expect(removedCount).toBe(2)
    // One page at the bulk cap, scoped like the recents list (cron/messaging/
    // subagent excluded), then the empty follow-up that ends the loop.
    expect(listAllProfileSessions).toHaveBeenCalledWith(
      500,
      1,
      'exclude',
      'recent',
      'default',
      expect.objectContaining({ excludeSources: expect.arrayContaining(['cron', 'subagent', 'tool']) })
    )
    expect(bulkDeleteSessions).toHaveBeenCalledTimes(1)
    expect(bulkDeleteSessions).toHaveBeenCalledWith(['s1', 's2'], 'default')
    expect($sessions.get()).toHaveLength(0)
    expect($pinnedSessionIds.get()).toEqual([])
  })

  it('groups ids by owning profile so each profile is deleted against its own db', async () => {
    const rows = [row('a1'), row('b1', { profile: 'work' }), row('a2')]

    listAllProfileSessions
      .mockResolvedValueOnce({ limit: 500, offset: 0, sessions: rows, total: rows.length })
      .mockResolvedValue(emptyPage)
    bulkDeleteSessions.mockImplementation((ids: string[]) => Promise.resolve({ deleted: ids.length, ok: true }))

    const { result } = renderHook(() => useSessionListActions({ profileScope: '__all__' }))

    await act(async () => {
      await result.current.clearAllSessions()
    })

    expect(bulkDeleteSessions).toHaveBeenCalledWith(['a1', 'a2'], 'default')
    expect(bulkDeleteSessions).toHaveBeenCalledWith(['b1'], 'work')
  })

  it('is a no-op (no delete calls) when the scope is already empty', async () => {
    listAllProfileSessions.mockResolvedValue(emptyPage)

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    let removedCount = -1

    await act(async () => {
      removedCount = await result.current.clearAllSessions()
    })

    expect(removedCount).toBe(0)
    expect(bulkDeleteSessions).not.toHaveBeenCalled()
  })

  it('still reconciles the list when a later batch rejects after earlier deletes succeeded', async () => {
    const first = row('s1')
    const second = row('s2')
    setSessions([first, second])
    setSessionsTotal(2)

    // Page 1 deletes fine; page 2's bulk call rejects mid-clear.
    listAllProfileSessions
      .mockResolvedValueOnce({ limit: 500, offset: 0, sessions: [first], total: 2 })
      .mockResolvedValueOnce({ limit: 500, offset: 0, sessions: [second], total: 1 })
    bulkDeleteSessions
      .mockResolvedValueOnce({ deleted: 1, ok: true })
      .mockRejectedValueOnce(new Error('backend down'))
    // The authoritative refresh reports what actually survived: s2.
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [second], total: 1, profile_totals: {} }))

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    await act(async () => {
      await expect(result.current.clearAllSessions()).rejects.toThrow('backend down')
    })

    // The failure propagates to the caller (the confirm dialog surfaces it),
    // but the finally-refresh still ran, reconciling the earlier optimistic
    // removals with the authoritative list instead of stranding them.
    expect(bulkDeleteSessions).toHaveBeenCalledTimes(2)
    expect(listSidebarSessions).toHaveBeenCalled()
    expect($sessions.get().map(s => s.id)).toEqual(['s2'])
  })
})
