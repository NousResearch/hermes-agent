import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $sessionsLimit, resetSessionsLimit, SIDEBAR_SESSIONS_PAGE_SIZE } from '@/store/layout'
import {
  $cronSessions,
  $freshDraftReady,
  $messagingSessions,
  $sessionProfilesTruncated,
  $sessions,
  $sessionsLoading,
  setCronSessions,
  setFreshDraftReady,
  setMessagingSessions,
  setSessionProfilesTruncated,
  setSessions,
  setSessionsLoading
} from '@/store/session'
import { $stalledSessionIds } from '@/store/session-states'

import { $gatewaySwitching, wipeSessionListsForGatewaySwitch } from './gateway-switch'

vi.mock('@/lib/query-client', () => ({
  invalidateProfileScopedQueries: vi.fn()
}))

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
})
