import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'

import { $checkInDue, $checkInReopenable, answerCheckIn, dismissCheckIn, reopenCheckIn } from './check-in'
import {
  bindDesktopMetrics,
  captureDesktopMetricsGate,
  DESKTOP_METRICS_STATE_KEY,
  resetDesktopMetricsForTests
} from './desktop-metrics'
import { immediateMetricsLocks, setMetricsGate } from './desktop-metrics-test-locks'
import { $guidanceLedger } from './guidance-ledger'
import { $shellChoice } from './shell-variant'
import { noteTaskFinished } from './starter-ideas'

const recorded = () =>
  Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '')
    .filter(key => key.startsWith(DESKTOP_METRICS_STATE_KEY))
    .map(key => window.localStorage.getItem(key))
    .join('')

beforeEach(() => {
  immediateMetricsLocks()
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'gateway', profile: 'default' })
  bindDesktopMetrics(null, 'gateway|default')
  $shellChoice.set(true)
  $guidanceLedger.set({ renameSplash: 'not_applicable' })
})

afterEach(() => {
  resetDesktopMetricsForTests()
  vi.unstubAllGlobals()
})

describe('the check-in (C7)', () => {
  it('is due after two chats in which a task finished, not one', () => {
    expect($checkInDue.get()).toBe(false)
    noteTaskFinished('chat-a')
    expect($checkInDue.get()).toBe(false)
    noteTaskFinished('chat-b')
    expect($checkInDue.get()).toBe(true)
  })

  it('is not for someone who used Hermes before', () => {
    $guidanceLedger.set({ renameSplash: 'seen', finishedTaskChats: ['a', 'b'] })

    expect($checkInDue.get()).toBe(false)
  })

  it('retires for good on an answer, with no way back from the tray', () => {
    noteTaskFinished('chat-a')
    noteTaskFinished('chat-b')
    answerCheckIn('going_well')

    expect($checkInDue.get()).toBe(false)
    expect($checkInReopenable.get()).toBe(false)
  })

  it('retires on Dismiss, comes back from the tray, and an answer then retires it', () => {
    noteTaskFinished('chat-a')
    noteTaskFinished('chat-b')
    dismissCheckIn()
    expect($checkInDue.get()).toBe(false)
    expect($checkInReopenable.get()).toBe(true)

    reopenCheckIn()
    expect($checkInDue.get()).toBe(true)

    answerCheckIn('could_be_better')
    expect($checkInDue.get()).toBe(false)
    expect($checkInReopenable.get()).toBe(false)
  })

  it('counts its answers as closed codes, behind consent', async () => {
    noteTaskFinished('chat-a')
    noteTaskFinished('chat-b')
    answerCheckIn('going_well')
    expect(recorded()).toBe('')

    await setMetricsGate(captureDesktopMetricsGate, 'on')
    answerCheckIn('could_be_better')
    dismissCheckIn()

    expect(recorded()).toContain('"ia.check_in.could_be_better|click":1')
    expect(recorded()).toContain('"ia.check_in.dismissed|click":1')
  })
})
