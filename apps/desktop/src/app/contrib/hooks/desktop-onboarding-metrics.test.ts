import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { bindDesktopMetrics, captureDesktopMetricsGate, resetDesktopMetricsForTests } from '@/store/desktop-metrics'
import { immediateMetricsLocks, setMetricsGate } from '@/store/desktop-metrics-test-locks'
import type { FreeTierSignInState } from '@/store/free-tier-sign-in'
import type { SharedMetricsRequester } from '@/store/shared-metrics'

import { signInTransition } from './desktop-onboarding-metrics'

const flush = () => new Promise(resolve => setTimeout(resolve, 0))

async function walk(...states: FreeTierSignInState[]): Promise<string[]> {
  const sent: string[] = []

  const request: SharedMetricsRequester = async (_method, params) => {
    sent.push(params?.step ? `${params.step}:${params.event}` : `${params?.signal}:${params?.target}`)

    // SAFETY: the metrics store fires and forgets (`void request(...).catch`); it never reads the answer.
    return undefined as never
  }

  bindDesktopMetrics(request)
  await setMetricsGate(captureDesktopMetricsGate, 'on')

  for (let i = 1; i < states.length; i++) {
    signInTransition(states[i - 1], states[i])
  }

  await flush()

  return sent
}

const CLOSED: FreeTierSignInState = { status: 'closed' }
const OFFER: FreeTierSignInState = { kind: 'trial', status: 'offer' }
const SETTING_UP: FreeTierSignInState = { minting: false, status: 'setting_up' }

beforeEach(() => {
  immediateMetricsLocks()
  window.localStorage.clear()
  resetDesktopMetricsForTests()
})

afterEach(() => {
  resetDesktopMetricsForTests()
  vi.unstubAllGlobals()
})

describe('sign-in funnel from the offer', () => {
  it('an offer the user signs in from and then cancels is a reach followed by a cancel', async () => {
    expect(await walk(CLOSED, OFFER, SETTING_UP, CLOSED)).toEqual(['sign_in:reached', 'cancelled:free_tier_sign_in'])
  })

  it('"Not now" on the offer is a reach, not a cancelled sign-in', async () => {
    expect(await walk(CLOSED, OFFER, CLOSED)).toEqual(['sign_in:reached'])
  })
})
