import { act, render, renderHook } from '@testing-library/react'
import { Suspense } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo, SidebarSessionsResponse } from '@/hermes'
import { $cronJobs, setCronJobs } from '@/store/cron'
import {
  $cronSessions,
  $messagingPlatformTotals,
  $messagingSessions,
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
const getCronJobs = vi.fn()

interface Deferred<T> {
  promise: Promise<T>
  resolve: (value: T) => void
}

/** Create a promise whose completion order the stale-response tests control. */
function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobs: (...args: unknown[]) => getCronJobs(...args),
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
  getCronJobs.mockReset()
  getCronJobs.mockResolvedValue([])
  listSidebarSessions.mockReset()
  listAllProfileSessions.mockReset()
  removed.ids = new Set()
  setCronJobs([])
  setSessions([])
  setCronSessions([])
  setMessagingSessions([])
  setMessagingPlatformTotals({})
  setMessagingTruncated(false)
  setSessionsLoading(false)
})

afterEach(() => {
  setCronJobs([])
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

  it('drops tombstoned rows from the messaging slice and per-platform paging too (#50928)', async () => {
    // The same delete race exists on every ingestion point: the batched
    // refresh's messaging slice and the per-platform "load more" pager must
    // both honor the tombstone, or a deleted platform thread resurrects.
    removed.ids = new Set(['tg-2'])
    listSidebarSessions.mockResolvedValue(
      sidebar({ sessions: [] }, [], [row('tg-1', { source: 'telegram' }), row('tg-2', { source: 'telegram' })])
    )

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))

    await act(async () => {
      await result.current.refreshSessions()
    })

    expect($messagingSessions.get().map(s => s.id)).toEqual(['tg-1'])

    // Per-platform pager: backend page still lists the doomed row.
    listAllProfileSessions.mockResolvedValue({
      sessions: [
        row('tg-1', { source: 'telegram' }),
        row('tg-2', { source: 'telegram' }),
        row('tg-3', { source: 'telegram' })
      ],
      total: 3
    })

    await act(async () => {
      await result.current.loadMoreMessagingForPlatform('telegram')
    })

    expect($messagingSessions.get().map(s => s.id)).toEqual(['tg-1', 'tg-3'])
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

  it('does not start a refresh callback captured before a profile switch', async () => {
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [] }))

    const { rerender, result } = renderHook(({ profileScope }) => useSessionListActions({ profileScope }), {
      initialProps: { profileScope: 'work' }
    })

    const staleRefresh = result.current.refreshSessions

    rerender({ profileScope: 'personal' })

    await act(async () => {
      await staleRefresh()
    })

    expect(listSidebarSessions).not.toHaveBeenCalled()
  })

  it('keeps the committed profile active when a later render is discarded', async () => {
    const never = new Promise<never>(() => undefined)
    let committedRefresh: (() => Promise<void>) | undefined

    /** Expose only callbacks from committed renders; suspended renders are discarded. */
    function Harness({ profileScope, suspend }: { profileScope: string; suspend: boolean }) {
      const actions = useSessionListActions({ profileScope })

      if (suspend) {
        throw never
      }

      committedRefresh = actions.refreshSessions

      return null
    }

    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [] }))

    const view = render(
      <Suspense fallback={null}>
        <Harness profileScope="work" suspend={false} />
      </Suspense>
    )

    const workRefresh = committedRefresh!

    view.rerender(
      <Suspense fallback={null}>
        <Harness profileScope="personal" suspend />
      </Suspense>
    )

    await act(async () => {
      await workRefresh()
    })

    expect(listSidebarSessions).toHaveBeenCalledWith(expect.objectContaining({ recentsProfile: 'work' }))
  })

  it('ignores an in-flight sidebar response after the active profile changes', async () => {
    const work = deferred<SidebarSessionsResponse>()
    const personal = deferred<SidebarSessionsResponse>()

    listSidebarSessions.mockImplementation(({ recentsProfile }) =>
      recentsProfile === 'work' ? work.promise : personal.promise
    )

    const { rerender, result } = renderHook(({ profileScope }) => useSessionListActions({ profileScope }), {
      initialProps: { profileScope: 'work' }
    })

    const workRefresh = result.current.refreshSessions()

    rerender({ profileScope: 'personal' })
    const personalRefresh = result.current.refreshSessions()

    await act(async () => {
      personal.resolve(
        sidebar(
          { sessions: [row('personal-session', { profile: 'personal' })] },
          [row('personal-cron', { profile: 'personal', source: 'cron' })],
          [row('personal-signal', { profile: 'personal', source: 'signal' })]
        )
      )
      await personalRefresh

      work.resolve(
        sidebar(
          { sessions: [row('work-session', { profile: 'work' })] },
          [row('work-cron', { profile: 'work', source: 'cron' })],
          [row('work-telegram', { profile: 'work', source: 'telegram' })]
        )
      )
      await workRefresh
    })

    expect($sessions.get().map(session => session.id)).toEqual(['personal-session'])
    expect($cronSessions.get().map(session => session.id)).toEqual(['personal-cron'])
    expect($messagingSessions.get().map(session => session.id)).toEqual(['personal-signal'])
  })

  it('scopes the cron-jobs fetch to the active profile', async () => {
    listSidebarSessions.mockResolvedValue(sidebar({ sessions: [] }))

    const scoped = renderHook(() => useSessionListActions({ profileScope: 'work' }))

    await act(async () => {
      await scoped.result.current.refreshCronJobs()
    })

    expect(getCronJobs).toHaveBeenLastCalledWith('work')
  })

  it('requests cron jobs for the unified scope', async () => {
    const unified = renderHook(() => useSessionListActions({ profileScope: '__all__' }))

    await act(async () => {
      await unified.result.current.refreshCronJobs()
    })

    expect(getCronJobs).toHaveBeenLastCalledWith('all')
  })

  it('ignores an out-of-order cron-jobs response from the previous profile', async () => {
    const work = deferred<Array<{ enabled: boolean; id: string }>>()
    const personal = deferred<Array<{ enabled: boolean; id: string }>>()

    getCronJobs.mockImplementation((profile: string) => (profile === 'work' ? work.promise : personal.promise))

    const { rerender, result } = renderHook(({ profileScope }) => useSessionListActions({ profileScope }), {
      initialProps: { profileScope: 'work' }
    })

    const workRefresh = result.current.refreshCronJobs()

    rerender({ profileScope: 'personal' })
    const personalRefresh = result.current.refreshCronJobs()

    await act(async () => {
      personal.resolve([{ enabled: true, id: 'personal-job' }])
      await personalRefresh

      work.resolve([{ enabled: true, id: 'work-job' }])
      await workRefresh
    })

    expect(getCronJobs.mock.calls.map(call => call[0])).toEqual(['work', 'personal'])
    expect($cronJobs.get().map(job => job.id)).toEqual(['personal-job'])
  })
})

describe('messaging profile scope', () => {
  it('refreshes messaging sessions only for the active profile', async () => {
    listAllProfileSessions.mockResolvedValue({
      sessions: [row('m1', { profile: 'work', source: 'signal' })],
      total: 1
    })
    const { result } = renderHook(() => useSessionListActions({ profileScope: 'work' }))

    await act(async () => {
      await result.current.refreshMessagingSessions()
    })

    expect(listAllProfileSessions).toHaveBeenCalledWith(
      expect.any(Number),
      1,
      'exclude',
      'recent',
      'work',
      expect.objectContaining({ excludeSources: expect.any(Array) })
    )
    expect($messagingSessions.get().map(s => s.id)).toEqual(['m1'])
  })

  it('keeps the explicit all-profiles view unified', async () => {
    listAllProfileSessions.mockResolvedValue({ sessions: [], total: 0 })
    const { result } = renderHook(() => useSessionListActions({ profileScope: '__all__' }))

    await act(async () => {
      await result.current.refreshMessagingSessions()
    })

    expect(listAllProfileSessions.mock.calls[0][4]).toBe('all')
  })

  it('keeps per-platform pagination on the active profile', async () => {
    setMessagingSessions([row('m1', { profile: 'work', source: 'signal' })])
    listAllProfileSessions.mockResolvedValue({
      sessions: [row('m1', { profile: 'work', source: 'signal' }), row('m2', { profile: 'work', source: 'signal' })],
      total: 2
    })
    const { result } = renderHook(() => useSessionListActions({ profileScope: 'work' }))

    await act(async () => {
      await result.current.loadMoreMessagingForPlatform('signal')
    })

    expect(listAllProfileSessions.mock.calls[0][4]).toBe('work')
    expect($messagingSessions.get().map(s => s.id)).toEqual(['m1', 'm2'])
    expect($messagingPlatformTotals.get()).toEqual({ 'work:signal': 2 })
  })

  it('keeps rows from every profile when paginating the unified scope', async () => {
    setMessagingSessions([row('work-signal', { profile: 'work', source: 'signal' })])
    listAllProfileSessions.mockResolvedValue({
      sessions: [
        row('work-signal', { profile: 'work', source: 'signal' }),
        row('personal-signal', { profile: 'personal', source: 'signal' })
      ],
      total: 2
    })

    const { result } = renderHook(() => useSessionListActions({ profileScope: '__all__' }))

    await act(async () => {
      await result.current.loadMoreMessagingForPlatform('signal')
    })

    expect(listAllProfileSessions.mock.calls[0][4]).toBe('all')
    expect($messagingSessions.get().map(session => session.id)).toEqual(['work-signal', 'personal-signal'])
    expect($messagingPlatformTotals.get()).toEqual({ 'all:signal': 2 })
  })

  it('keeps loaded platform rows when pagination fails', async () => {
    const loaded = [row('work-signal', { profile: 'work', source: 'signal' })]
    setMessagingSessions(loaded)
    setMessagingPlatformTotals({ 'work:signal': 12 })
    listAllProfileSessions.mockRejectedValue(new Error('request failed'))

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'work' }))

    await act(async () => {
      await expect(result.current.loadMoreMessagingForPlatform('signal')).resolves.toBeUndefined()
    })

    expect($messagingSessions.get()).toEqual(loaded)
    expect($messagingPlatformTotals.get()).toEqual({ 'work:signal': 12 })
  })

  it('keeps resolved platform totals separate across profile switches', async () => {
    listAllProfileSessions.mockResolvedValue({
      sessions: [row('work-signal', { profile: 'work', source: 'signal' })],
      total: 42
    })

    const { rerender, result } = renderHook(({ profileScope }) => useSessionListActions({ profileScope }), {
      initialProps: { profileScope: 'work' }
    })

    await act(async () => {
      await result.current.loadMoreMessagingForPlatform('signal')
    })

    expect($messagingPlatformTotals.get()).toEqual({ 'work:signal': 42 })

    rerender({ profileScope: 'personal' })

    expect($messagingPlatformTotals.get()['personal:signal']).toBeUndefined()
    expect($messagingPlatformTotals.get()['work:signal']).toBe(42)

    listAllProfileSessions.mockResolvedValue({
      sessions: [row('personal-signal', { profile: 'personal', source: 'signal' })],
      total: 3
    })

    await act(async () => {
      await result.current.loadMoreMessagingForPlatform('signal')
    })

    expect($messagingPlatformTotals.get()).toEqual({ 'personal:signal': 3, 'work:signal': 42 })

    rerender({ profileScope: 'work' })

    expect($messagingPlatformTotals.get()['work:signal']).toBe(42)
  })

  it('ignores an older overlapping load-more response for the same profile and platform', async () => {
    const older = deferred<{ sessions: SessionInfo[]; total: number }>()
    const newer = deferred<{ sessions: SessionInfo[]; total: number }>()

    setMessagingSessions([row('m1', { profile: 'work', source: 'signal' })])
    listAllProfileSessions.mockImplementationOnce(() => older.promise).mockImplementationOnce(() => newer.promise)

    const { result } = renderHook(() => useSessionListActions({ profileScope: 'work' }))
    const olderLoad = result.current.loadMoreMessagingForPlatform('signal')

    setMessagingSessions([
      row('m1', { profile: 'work', source: 'signal' }),
      row('m2', { profile: 'work', source: 'signal' })
    ])
    const newerLoad = result.current.loadMoreMessagingForPlatform('signal')

    await act(async () => {
      newer.resolve({
        sessions: [
          row('m1', { profile: 'work', source: 'signal' }),
          row('m2', { profile: 'work', source: 'signal' }),
          row('m3', { profile: 'work', source: 'signal' })
        ],
        total: 3
      })
      await newerLoad

      older.resolve({
        sessions: [row('m1', { profile: 'work', source: 'signal' }), row('m2', { profile: 'work', source: 'signal' })],
        total: 2
      })
      await olderLoad
    })

    expect($messagingSessions.get().map(session => session.id)).toEqual(['m1', 'm2', 'm3'])
    expect($messagingPlatformTotals.get()).toEqual({ 'work:signal': 3 })
  })

  it('ignores an in-flight response after the active profile changes', async () => {
    const work = deferred<{ sessions: SessionInfo[]; total: number }>()
    const personal = deferred<{ sessions: SessionInfo[]; total: number }>()

    listAllProfileSessions.mockImplementation((_limit, _min, _archived, _order, profile) =>
      profile === 'work' ? work.promise : personal.promise
    )

    const { rerender, result } = renderHook(({ profileScope }) => useSessionListActions({ profileScope }), {
      initialProps: { profileScope: 'work' }
    })

    const workRefresh = result.current.refreshMessagingSessions()
    rerender({ profileScope: 'personal' })
    const personalRefresh = result.current.refreshMessagingSessions()

    await act(async () => {
      personal.resolve({
        sessions: [row('personal-message', { profile: 'personal', source: 'telegram' })],
        total: 1
      })
      await personalRefresh
      work.resolve({ sessions: [row('work-message', { profile: 'work', source: 'signal' })], total: 1 })
      await workRefresh
    })

    expect(listAllProfileSessions.mock.calls.map(call => call[4])).toEqual(['work', 'personal'])
    expect($messagingSessions.get().map(session => session.id)).toEqual(['personal-message'])
  })

  it('does not let a callback captured before a profile switch disturb current totals', async () => {
    listAllProfileSessions.mockResolvedValue({ sessions: [], total: 0 })
    setMessagingPlatformTotals({ 'work:signal': 12 })

    const { rerender, result } = renderHook(({ profileScope }) => useSessionListActions({ profileScope }), {
      initialProps: { profileScope: 'work' }
    })

    const staleRefresh = result.current.refreshMessagingSessions

    rerender({ profileScope: 'personal' })

    await act(async () => {
      await staleRefresh()
    })

    expect(listAllProfileSessions).not.toHaveBeenCalled()
    expect($messagingPlatformTotals.get()).toEqual({ 'work:signal': 12 })
  })
})
