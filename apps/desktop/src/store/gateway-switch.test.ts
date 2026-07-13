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
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { $gatewaySwitching, wipeSessionListsForGatewaySwitch } from './gateway-switch'

vi.mock('@/lib/query-client', () => ({
  queryClient: { invalidateQueries: vi.fn() }
}))

describe('wipeSessionListsForGatewaySwitch', () => {
  beforeEach(() => {
    $gatewaySwitching.set(false)
    setSessions([{ id: 's1', title: 'old', profile: 'default' } as never])
    setSessionsTotal(1)
    setCronSessions([{ id: 'c1', title: 'cron', profile: 'default' } as never])
    setMessagingSessions([{ id: 'm1', title: 'tg', profile: 'default' } as never])
    setSessionsLoading(false)
    setFreshDraftReady(false)
    $sessionsLimit.set(SIDEBAR_SESSIONS_PAGE_SIZE * 3)
    upsertSubagent('s1', { goal: 'old worker', status: 'running', subagent_id: 'worker-1', task_index: 0 })
  })

  afterEach(() => {
    resetSessionsLimit()
    setSessions([])
    setCronSessions([])
    setMessagingSessions([])
    $subagentsBySession.set({})
    setSessionsLoading(true)
    $gatewaySwitching.set(false)
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
    expect($subagentsBySession.get()).toEqual({})
  })
})
