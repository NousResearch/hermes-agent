// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { pageStorage } from '@/test/page-storage'

import {
  $desktopMetricsGate,
  bindDesktopMetrics,
  captureDesktopMetricsGate,
  resetDesktopMetricsForTests
} from './desktop-metrics'
import { immediateMetricsLocks, setMetricsGate } from './desktop-metrics-test-locks'
import {
  $sharedMetricsConsent,
  $sharedMetricsDetailsOpen,
  answerSharedMetricsOffer,
  type SharedMetricsConsent
} from './shared-metrics'

beforeEach(() => {
  immediateMetricsLocks()
  pageStorage().clear()
  resetDesktopMetricsForTests()
  $sharedMetricsConsent.set(null)
  $sharedMetricsDetailsOpen.set(false)
})

afterEach(() => {
  resetDesktopMetricsForTests()
  $sharedMetricsConsent.set(null)
  $sharedMetricsDetailsOpen.set(false)
  vi.unstubAllGlobals()
})

describe('first-run consent ownership', () => {
  it('does not answer or close the replacement profile offer when an older opt-in finishes', async () => {
    const saved = Promise.withResolvers<SharedMetricsConsent>()
    const sourceRequest = vi.fn().mockReturnValue(saved.promise)
    const targetRequest = vi.fn().mockResolvedValue({ recorded: true })
    const unanswered = { enabled: false, send: false, decided: false }

    bindDesktopMetrics(null, 'local|alpha')
    await setMetricsGate(captureDesktopMetricsGate, 'off', false)
    const answer = answerSharedMetricsOffer(sourceRequest, 'share', 'local|alpha')

    await vi.waitFor(() => expect(sourceRequest).toHaveBeenCalledTimes(1))
    bindDesktopMetrics(targetRequest, 'local|beta')
    await setMetricsGate(captureDesktopMetricsGate, 'off', false)
    $sharedMetricsConsent.set(unanswered)
    $sharedMetricsDetailsOpen.set(true)
    saved.resolve({ enabled: true, send: true, decided: true })
    await answer

    expect($desktopMetricsGate.get()).toBe('off')
    expect($sharedMetricsConsent.get()).toEqual(unanswered)
    expect($sharedMetricsDetailsOpen.get()).toBe(true)
    expect(targetRequest).not.toHaveBeenCalled()
  })
})
