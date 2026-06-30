import { afterEach, describe, expect, it } from 'vitest'

import {
  $pendingPlansBySession,
  $planModeEnabled,
  clearPendingPlan,
  markPendingPlanReady,
  migratePendingPlan,
  PLAN_PENDING_SESSION,
  setPendingPlan,
  setPlanModeEnabled
} from './plans'

describe('plan store', () => {
  afterEach(() => {
    $planModeEnabled.set(false)
    $pendingPlansBySession.set({})
  })

  it('toggles plan mode', () => {
    setPlanModeEnabled(true)
    expect($planModeEnabled.get()).toBe(true)

    setPlanModeEnabled(false)
    expect($planModeEnabled.get()).toBe(false)
  })

  it('marks pending plans ready', () => {
    setPendingPlan('s1', { attachments: [], originalText: 'ship it', state: 'planning' })

    markPendingPlanReady('s1')

    expect($pendingPlansBySession.get().s1).toMatchObject({
      originalText: 'ship it',
      state: 'ready'
    })
  })

  it('migrates a pre-session pending plan once a runtime session exists', () => {
    setPendingPlan(PLAN_PENDING_SESSION, { attachments: [], originalText: 'plan this', state: 'planning' })

    migratePendingPlan(PLAN_PENDING_SESSION, 'runtime-1')

    expect($pendingPlansBySession.get()[PLAN_PENDING_SESSION]).toBeUndefined()
    expect($pendingPlansBySession.get()['runtime-1']).toMatchObject({
      originalText: 'plan this',
      state: 'planning'
    })
  })

  it('clears pending plans', () => {
    setPendingPlan('runtime-1', { attachments: [], originalText: 'plan this', state: 'ready' })

    clearPendingPlan('runtime-1')

    expect($pendingPlansBySession.get()['runtime-1']).toBeUndefined()
  })
})
