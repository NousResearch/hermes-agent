import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $sessionsLimit, resetSessionsLimit, SIDEBAR_SESSIONS_PAGE_SIZE } from '@/store/layout'
import {
  $activeProjectId,
  $projects,
  $projectScope,
  $projectsRpcAvailable,
  $projectTree,
  $removedSessionIds,
  $reposScanning,
  ALL_PROJECTS,
  enterProject
} from '@/store/projects'
import {
  $activeSessionId,
  $cronSessions,
  $freshDraftReady,
  $messagingSessions,
  $sessionProfilesTruncated,
  $sessions,
  $sessionsLoading,
  setActiveSessionId,
  setCronSessions,
  setFreshDraftReady,
  setMessagingSessions,
  setSessionProfilesTruncated,
  setSessions,
  setSessionsLoading
} from '@/store/session'
import { $stalledSessionIds } from '@/store/session-states'

import {
  $gatewaySwitching,
  beginGatewaySwitch,
  endGatewaySwitch,
  recoverActiveSourceAfterFailedGatewaySwitch,
  registerGatewaySwitchLifecycle,
  wipeSessionListsForGatewaySwitch
} from './gateway-switch'

vi.mock('@/lib/query-client', () => ({
  invalidateProfileScopedQueries: vi.fn()
}))

vi.mock(import('@/store/profile'), async importOriginal => {
  const actual = await importOriginal()

  return {
    ...actual,
    invalidateProfileListFetches: vi.fn()
  }
})

const { invalidateProfileListFetches } = await import('@/store/profile')

describe('wipeSessionListsForGatewaySwitch', () => {
  beforeEach(() => {
    $gatewaySwitching.set(false)
    setSessions([{ id: 's1', title: 'old', profile: 'default' } as never])
    setSessionProfilesTruncated({ default: true })
    setCronSessions([{ id: 'c1', title: 'cron', profile: 'default' } as never])
    setMessagingSessions([{ id: 'm1', title: 'tg', profile: 'default' } as never])
    $stalledSessionIds.set(['s1'])
    setSessionsLoading(false)
    setFreshDraftReady(false)
    $sessionsLimit.set(SIDEBAR_SESSIONS_PAGE_SIZE * 3)
    $projects.set([{ id: 'p_old', label: 'old', repos: [], isNoProject: false } as never])
    $projectTree.set([{ id: 'p_old', label: 'old', repos: [], sessionCount: 0 } as never])
    $activeProjectId.set('p_old')
    enterProject('p_old')
    $projectsRpcAvailable.set(true)
    $reposScanning.set(true)
    $removedSessionIds.set(new Set(['s1']))
  })

  afterEach(() => {
    resetSessionsLimit()
    setSessions([])
    setCronSessions([])
    setMessagingSessions([])
    $stalledSessionIds.set([])
    setSessionsLoading(true)
    $gatewaySwitching.set(false)
  })

  it('clears lists and arms loading so sidebar skeletons retrigger', () => {
    wipeSessionListsForGatewaySwitch()

    expect($sessions.get()).toEqual([])
    expect($sessionProfilesTruncated.get()).toEqual({})
    expect($cronSessions.get()).toEqual([])
    expect($messagingSessions.get()).toEqual([])
    expect($stalledSessionIds.get()).toEqual([])
    expect($sessionsLoading.get()).toBe(true)
    expect($sessionsLimit.get()).toBe(SIDEBAR_SESSIONS_PAGE_SIZE)
    expect($freshDraftReady.get()).toBe(true)
  })

  it('strands in-flight profile-list fetches so the old backend cannot repaint the rail (#85731)', () => {
    // The soft re-home moves /api/profiles routing to the NEW backend; a
    // response still in flight from the previous one must be invalidated
    // here, in the same wipe every connection/mode apply funnels through.
    wipeSessionListsForGatewaySwitch()

    expect(invalidateProfileListFetches).toHaveBeenCalled()
  })

  it('clears per-profile project state so the next gateway repaints its own', () => {
    wipeSessionListsForGatewaySwitch()

    expect($projects.get()).toEqual([])
    expect($projectTree.get()).toEqual([])
    expect($activeProjectId.get()).toBeNull()
    expect($projectScope.get()).toBe(ALL_PROJECTS)
    expect($projectsRpcAvailable.get()).toBeNull()
    expect($reposScanning.get()).toBe(false)
    expect($removedSessionIds.get().size).toBe(0)
  })
})

describe('beginGatewaySwitch / endGatewaySwitch — the shared switch commit point (#93937)', () => {
  beforeEach(() => {
    $gatewaySwitching.set(false)
    setSessions([{ id: 's1', title: 'old', profile: 'default' } as never])
    setActiveSessionId('a93bb39d')
    setSessionsLoading(false)
  })

  afterEach(() => {
    setSessions([])
    setActiveSessionId(null)
    setSessionsLoading(true)
    $gatewaySwitching.set(false)
  })

  it('raises the barrier, runs the registered machine-context reset, then wipes — synchronously, in that order', () => {
    const seen: string[] = []

    const off = registerGatewaySwitchLifecycle({
      beforeConnectionSwitch: () => {
        // The reset runs behind the barrier and BEFORE the wipe: it may still
        // read the outgoing session (to fresh-draft it), never a half-wiped one.
        seen.push(
          `switching=${$gatewaySwitching.get()} active=${$activeSessionId.get()} rows=${$sessions.get().length}`
        )
      },
      refreshSessions: async () => undefined
    })

    beginGatewaySwitch()

    expect(seen).toEqual(['switching=true active=a93bb39d rows=1'])
    expect($gatewaySwitching.get()).toBe(true)
    // The previous backend's runtime binding is gone before anything can dial.
    expect($activeSessionId.get()).toBeNull()
    expect($sessions.get()).toEqual([])
    expect($sessionsLoading.get()).toBe(true)

    endGatewaySwitch()
    expect($gatewaySwitching.get()).toBe(false)
    off()
  })

  it('tears down its barrier when the registered lifecycle throws before the wipe', () => {
    const failure = new Error('machine-context reset failed')

    const off = registerGatewaySwitchLifecycle({
      beforeConnectionSwitch: () => {
        throw failure
      },
      refreshSessions: vi.fn(async () => undefined)
    })

    expect(() => beginGatewaySwitch()).toThrow(failure)
    expect($gatewaySwitching.get()).toBe(false)
    // No wipe started, so the still-active source remains intact and needs no
    // repaint. A later switch can acquire and release barrier ownership.
    expect($activeSessionId.get()).toBe('a93bb39d')
    expect($sessions.get()).toHaveLength(1)

    off()
    const next = beginGatewaySwitch()
    expect($gatewaySwitching.get()).toBe(true)
    endGatewaySwitch(next)
    expect($gatewaySwitching.get()).toBe(false)
  })

  it('does not disarm a newer switch while a partial-wipe recovery refresh is pending', async () => {
    const failure = new Error('profile fetch invalidation failed')
    let finishRefresh: () => void = () => undefined
    let refreshCompleted = false

    const refreshPending = new Promise<void>(resolve => {
      finishRefresh = resolve
    })

    const refreshSessions = vi.fn(async () => {
      await refreshPending
      refreshCompleted = true
    })

    const off = registerGatewaySwitchLifecycle({ beforeConnectionSwitch: () => undefined, refreshSessions })

    vi.mocked(invalidateProfileListFetches).mockImplementationOnce(() => {
      throw failure
    })

    expect(() => beginGatewaySwitch()).toThrow(failure)
    expect($gatewaySwitching.get()).toBe(false)
    await vi.waitFor(() => expect(refreshSessions).toHaveBeenCalledTimes(1))

    // A newer switch takes ownership while the old source repaint is still in
    // flight. Completing the old repaint must not lower the newer skeleton.
    const next = beginGatewaySwitch()
    expect($gatewaySwitching.get()).toBe(true)
    expect($sessionsLoading.get()).toBe(true)

    finishRefresh()
    await vi.waitFor(() => expect(refreshCompleted).toBe(true))
    await Promise.resolve()

    expect($sessionsLoading.get()).toBe(true)
    expect($gatewaySwitching.get()).toBe(true)

    endGatewaySwitch(next)
    expect($gatewaySwitching.get()).toBe(false)
    off()
  })

  it('does not refresh through a newer route when recovery is superseded before it starts', async () => {
    const failure = new Error('profile fetch invalidation failed')
    const refreshSessions = vi.fn(async () => undefined)
    const off = registerGatewaySwitchLifecycle({ beforeConnectionSwitch: () => undefined, refreshSessions })

    vi.mocked(invalidateProfileListFetches).mockImplementationOnce(() => {
      throw failure
    })

    expect(() => beginGatewaySwitch()).toThrow(failure)

    // Recovery is queued on a microtask. A newer switch that starts first owns
    // the active route, so the stale recovery must not issue a request through it.
    const next = beginGatewaySwitch()
    await Promise.resolve()
    await Promise.resolve()

    expect(refreshSessions).not.toHaveBeenCalled()
    expect($sessionsLoading.get()).toBe(true)

    endGatewaySwitch(next)
    off()
  })

  it('the barrier belongs to the LATEST switch: an older switch ending mid-commit of a newer one is a no-op', () => {
    const older = beginGatewaySwitch()
    const newer = beginGatewaySwitch()

    endGatewaySwitch(older)
    expect($gatewaySwitching.get()).toBe(true)

    endGatewaySwitch(newer)
    expect($gatewaySwitching.get()).toBe(false)

    // Host teardown forces it down regardless of ownership.
    beginGatewaySwitch()
    endGatewaySwitch()
    expect($gatewaySwitching.get()).toBe(false)
  })

  it('repeated overlapping commits safely re-run the lifecycle and wipe without losing barrier ownership', () => {
    const beforeConnectionSwitch = vi.fn()

    const off = registerGatewaySwitchLifecycle({
      beforeConnectionSwitch,
      refreshSessions: async () => undefined
    })

    const older = beginGatewaySwitch()
    // Simulate stale gateway-bound state racing back before the newer commit.
    setSessions([{ id: 'late', title: 'late old-source row', profile: 'default' } as never])
    setActiveSessionId('late-runtime')
    const newer = beginGatewaySwitch()

    expect(beforeConnectionSwitch).toHaveBeenCalledTimes(2)
    expect($sessions.get()).toEqual([])
    expect($activeSessionId.get()).toBeNull()

    endGatewaySwitch(older)
    expect($gatewaySwitching.get()).toBe(true)
    endGatewaySwitch(newer)
    expect($gatewaySwitching.get()).toBe(false)
    off()
  })

  it('still severs the bindings when no lifecycle is registered (windows that never mount the boot hook)', () => {
    beginGatewaySwitch()

    expect($gatewaySwitching.get()).toBe(true)
    expect($activeSessionId.get()).toBeNull()
    expect($sessions.get()).toEqual([])

    endGatewaySwitch()
    expect($gatewaySwitching.get()).toBe(false)
  })

  it('an unregistered lifecycle no longer runs, and a stale unregister cannot evict a newer one', () => {
    const first = vi.fn()
    const second = vi.fn()

    const offFirst = registerGatewaySwitchLifecycle({
      beforeConnectionSwitch: first,
      refreshSessions: async () => undefined
    })

    const offSecond = registerGatewaySwitchLifecycle({
      beforeConnectionSwitch: second,
      refreshSessions: async () => undefined
    })

    // Stale unregister from the older host: the newer registration stays.
    offFirst()
    beginGatewaySwitch()
    endGatewaySwitch()

    expect(first).not.toHaveBeenCalled()
    expect(second).toHaveBeenCalledTimes(1)

    offSecond()
    beginGatewaySwitch()
    endGatewaySwitch()

    expect(second).toHaveBeenCalledTimes(1)
  })

  it('recoverActiveSourceAfterFailedGatewaySwitch re-pulls the still-active source and disarms the skeleton', async () => {
    const refreshSessions = vi.fn(async () => undefined)
    const off = registerGatewaySwitchLifecycle({ beforeConnectionSwitch: () => undefined, refreshSessions })

    const token = beginGatewaySwitch()
    expect($sessionsLoading.get()).toBe(true)

    recoverActiveSourceAfterFailedGatewaySwitch(token)
    endGatewaySwitch(token)
    await vi.waitFor(() => expect($sessionsLoading.get()).toBe(false))

    expect(refreshSessions).toHaveBeenCalledTimes(1)
    off()
  })

  it('a failing repaint (or none registered) still disarms the skeleton', async () => {
    const off = registerGatewaySwitchLifecycle({
      beforeConnectionSwitch: () => undefined,
      refreshSessions: async () => {
        throw new Error('backend busy')
      }
    })

    setSessionsLoading(true)
    const failingToken = beginGatewaySwitch()
    recoverActiveSourceAfterFailedGatewaySwitch(failingToken)
    endGatewaySwitch(failingToken)
    await vi.waitFor(() => expect($sessionsLoading.get()).toBe(false))
    off()

    setSessionsLoading(true)
    const debug = vi.spyOn(console, 'debug').mockImplementation(() => undefined)
    const missingLifecycleToken = beginGatewaySwitch()

    recoverActiveSourceAfterFailedGatewaySwitch(missingLifecycleToken)
    endGatewaySwitch(missingLifecycleToken)
    await vi.waitFor(() => expect($sessionsLoading.get()).toBe(false))

    expect(debug).toHaveBeenCalledWith(
      '[gateway-switch] cannot repaint the active source because no switch lifecycle is registered'
    )
    debug.mockRestore()
  })
})
