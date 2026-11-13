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

  it.each(['share', 'off'] as const)('applies a %s answer after a same-profile focus refresh', async choice => {
    const read = Promise.withResolvers<SharedMetricsConsent>()
    const saved = Promise.withResolvers<SharedMetricsConsent>()
    const answered = { enabled: choice === 'share', send: choice === 'share', decided: true }
    const unanswered = { enabled: false, send: false, decided: false }

    const request = vi.fn().mockImplementation(method =>
      method === 'shared_metrics.set' ? saved.promise : Promise.resolve(answered))

    bindDesktopMetrics(vi.fn().mockResolvedValue({ recorded: true }), 'local|alpha')
    await setMetricsGate(captureDesktopMetricsGate, 'off', false)
    $sharedMetricsConsent.set(unanswered)
    $sharedMetricsDetailsOpen.set(true)

    const refresh = captureDesktopMetricsGate('local|alpha').then(async applyGate => {
      const consent = await read.promise

      await applyGate(consent.enabled ? 'on' : 'off', consent.decided)
    })

    const answer = answerSharedMetricsOffer(request, choice, 'local|alpha')

    await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    read.resolve(unanswered)
    await refresh
    saved.resolve(answered)
    await answer

    expect($sharedMetricsConsent.get()).toEqual(answered)
    expect($desktopMetricsGate.get()).toBe(choice === 'share' ? 'on' : 'off')
    expect($sharedMetricsDetailsOpen.get()).toBe(false)
  })

  it.each([true, false])('preserves newer off consent (decided=%s) instead of replaying a saved opt-in', async decided => {
    const saved = Promise.withResolvers<SharedMetricsConsent>()
    const withdrawn = { enabled: false, send: false, decided }

    const request = vi.fn().mockImplementation(method =>
      method === 'shared_metrics.set' ? saved.promise : Promise.resolve(withdrawn))

    bindDesktopMetrics(vi.fn().mockResolvedValue({ recorded: true }), 'local|alpha')
    await setMetricsGate(captureDesktopMetricsGate, 'off', false)
    const answer = answerSharedMetricsOffer(request, 'share', 'local|alpha')

    await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    await setMetricsGate(captureDesktopMetricsGate, 'off', decided)
    $sharedMetricsConsent.set(withdrawn)
    $sharedMetricsDetailsOpen.set(!decided)
    saved.resolve({ enabled: true, send: true, decided: true })
    await answer

    expect($desktopMetricsGate.get()).toBe('off')
    expect($sharedMetricsConsent.get()).toEqual(withdrawn)
    expect($sharedMetricsDetailsOpen.get()).toBe(!decided)
  })
})
