import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $sessionsLimit, resetSessionsLimit, SIDEBAR_SESSIONS_PAGE_SIZE } from '@/store/layout'
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

    beginGatewaySwitch()
    expect($sessionsLoading.get()).toBe(true)

    recoverActiveSourceAfterFailedGatewaySwitch()
    endGatewaySwitch()
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
    recoverActiveSourceAfterFailedGatewaySwitch()
    await vi.waitFor(() => expect($sessionsLoading.get()).toBe(false))
    off()

    setSessionsLoading(true)
    recoverActiveSourceAfterFailedGatewaySwitch()
    await vi.waitFor(() => expect($sessionsLoading.get()).toBe(false))
  })
})
