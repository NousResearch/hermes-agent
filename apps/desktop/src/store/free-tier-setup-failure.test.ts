import { afterEach, describe, expect, it } from 'vitest'

import {
  $freeTierStatus,
  type FreeTierRequester,
  freeTierSetupFailure,
  provisionFreeTier
} from '@/store/free-tier'
import type { FreeTierStatus } from '@/types/hermes'

const NO_IDENTITY: FreeTierStatus = {
  available: false,
  enabled: true,
  has_guest: false,
  label: 'Nous · free tier',
  model: 'nous/welcome',
  notice_pending: false
}

afterEach(() => {
  $freeTierStatus.set(null)
})

describe('freeTierSetupFailure', () => {
  it('is nothing when the backend reported no failure, the tier is off, or an identity exists', () => {
    expect(freeTierSetupFailure(null)).toBeNull()
    expect(freeTierSetupFailure(NO_IDENTITY)).toBeNull()
    expect(freeTierSetupFailure({ ...NO_IDENTITY, enabled: false, error_code: 'anon_unreachable' })).toBeNull()
    expect(freeTierSetupFailure({ ...NO_IDENTITY, has_guest: true, error_code: 'anon_unreachable' })).toBeNull()
  })

  it('carries the code, the sentence, and the wait', () => {
    expect(
      freeTierSetupFailure({
        ...NO_IDENTITY,
        error: 'Lots of people are getting started right now.',
        error_code: 'anon_rate_limited',
        retry_after: 41.6,
        retryable: true
      })
    ).toEqual({
      code: 'anon_rate_limited',
      door: 'sign_in',
      message: 'Lots of people are getting started right now.',
      retryAfter: 42,
      retryable: true
    })
  })

  it.each([
    ['anon_unreachable', 'retry'],
    ['anon_gate_closed', 'sign_in'],
    ['something_newer', 'sign_in']
  ])('%s opens the %s door', (code, door) => {
    // A sign-in goes through the same service that just refused: only offer
    // it when that service answered at all.
    expect(freeTierSetupFailure({ ...NO_IDENTITY, error_code: code })?.door).toBe(door)
  })

  it('treats a missing retryable flag as not retryable', () => {
    expect(freeTierSetupFailure({ ...NO_IDENTITY, error_code: 'anon_gate_closed' })?.retryable).toBe(false)
  })
})

describe('provisionFreeTier', () => {
  it('asks the backend to try again, then re-reads the verdict', async () => {
    const calls: string[] = []
    const after: FreeTierStatus = { ...NO_IDENTITY, available: true, has_guest: true }

    const requestGateway = (async <T>(method: string): Promise<T> => {
      calls.push(method)

      return (method === 'free_tier.status' ? after : { has_guest: true, enabled: true }) as T
    }) satisfies FreeTierRequester

    expect(await provisionFreeTier(requestGateway)).toEqual(after)
    expect(calls).toEqual(['free_tier.provision', 'free_tier.status'])
    expect($freeTierStatus.get()).toEqual(after)
  })

  it('still reports what the backend knows when the retry call itself fails', async () => {
    const failed: FreeTierStatus = { ...NO_IDENTITY, error_code: 'anon_unreachable', retryable: true, retry_after: 15 }

    const requestGateway = (async <T>(method: string): Promise<T> => {
      if (method === 'free_tier.provision') {
        throw new Error('gateway away')
      }

      return failed as T
    }) satisfies FreeTierRequester

    expect(await provisionFreeTier(requestGateway)).toEqual(failed)
  })
})
