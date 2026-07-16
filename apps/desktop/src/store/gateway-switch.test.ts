import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $sessionsLimit, resetSessionsLimit, SIDEBAR_SESSIONS_PAGE_SIZE } from '@/store/layout'
import {
  $cronSessions,
  $freshDraftReady,
  $messagingSessions,
  $sessions,
  $sessionsLoading,
  $sessionsTotal,
  setCronSessions,
  setFreshDraftReady,
  setMessagingSessions,
  setSessions,
  setSessionsLoading,
  setSessionsTotal
} from '@/store/session'

import { $gatewaySwitching, wipeSessionListsForGatewaySwitch } from './gateway-switch'

const requestFreshSession = vi.fn()

vi.mock('@/lib/query-client', () => ({
  invalidateProfileScopedQueries: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => {
  const actual = await importOriginal<typeof import('@/store/profile')>()

  return {
    ...actual,
    requestFreshSession: (...args: unknown[]) => requestFreshSession(...args)
  }
})

vi.mock('@/app/routes', () => ({
  routeSessionId: (pathname: string) => {
    if (!pathname || pathname === '/' || pathname.startsWith('/settings')) {
      return null
    }

    const id = pathname.replace(/^\//, '')

    return id && !id.includes('/') ? id : null
  }
}))

describe('wipeSessionListsForGatewaySwitch', () => {
  beforeEach(() => {
    $gatewaySwitching.set(false)
    requestFreshSession.mockReset()
    window.location.hash = ''
    setSessions([{ id: 's1', title: 'old', profile: 'default' } as never])
    setSessionsTotal(1)
    setCronSessions([{ id: 'c1', title: 'cron', profile: 'default' } as never])
    setMessagingSessions([{ id: 'm1', title: 'tg', profile: 'default' } as never])
    setSessionsLoading(false)
    setFreshDraftReady(false)
    $sessionsLimit.set(SIDEBAR_SESSIONS_PAGE_SIZE * 3)
  })

  afterEach(() => {
    resetSessionsLimit()
    setSessions([])
    setCronSessions([])
    setMessagingSessions([])
    setSessionsLoading(true)
    $gatewaySwitching.set(false)
    window.location.hash = ''
  })

  it('clears lists and arms loading so sidebar skeletons retrigger', () => {
    wipeSessionListsForGatewaySwitch()

    expect($sessions.get()).toEqual([])
    expect($sessionsTotal.get()).toBe(0)
    expect($cronSessions.get()).toEqual([])
    expect($messagingSessions.get()).toEqual([])
    expect($sessionsLoading.get()).toBe(true)
    expect($sessionsLimit.get()).toBe(SIDEBAR_SESSIONS_PAGE_SIZE)
    expect($freshDraftReady.get()).toBe(true)
    expect(requestFreshSession).not.toHaveBeenCalled()
  })

  it('requests a blank draft when the URL still points at a previous-gateway session', () => {
    window.location.hash = '#/old-session-id'

    wipeSessionListsForGatewaySwitch()

    expect(requestFreshSession).toHaveBeenCalledTimes(1)
  })

  it('does not close Settings / overlay routes during a soft switch', () => {
    window.location.hash = '#/settings?tab=gateway'

    wipeSessionListsForGatewaySwitch()

    expect(requestFreshSession).not.toHaveBeenCalled()
  })
})
