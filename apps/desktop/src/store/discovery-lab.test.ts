import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { signUpHref } from '@/lib/visitor-access'

import { $discoveryLabUsageStatsConsent, discoveryLabVisitorAccess, setDiscoveryLabSignUp } from './discovery-lab'
import { $guidanceLedger } from './guidance-ledger'
import {
  $sharedMetricsConsent,
  $sharedMetricsDetailsOpen,
  $sharedMetricsOfferConsent,
  answerSharedMetricsOffer
} from './shared-metrics'
import { $shellChoice } from './shell-variant'
import { $usageStatsOfferPlace } from './usage-stats-offer'
import { $signInFailed, $signUpRequired, $visitorAccess, installVisitorAccess } from './visitor-access'

const UNANSWERED = { decided: false, enabled: false, send: false }
const ANSWERED = { decided: true, enabled: false, send: false }

beforeEach(() => {
  window.localStorage.clear()
  window.sessionStorage.clear()
  $shellChoice.set(true)
  $guidanceLedger.set({})
  $visitorAccess.set({ kind: 'open' })
  $signInFailed.set(false)
  $sharedMetricsConsent.set(null)
  $sharedMetricsDetailsOpen.set(false)
  $discoveryLabUsageStatsConsent.set(null)
})

afterEach(() => {
  window.history.replaceState(null, '', '/')
})

describe('the lab untouched', () => {
  it('leaves visitor access to the page: open, as every page is today', () => {
    window.history.replaceState(null, '', '/?hermes_sign_in=failed#/')

    expect(discoveryLabVisitorAccess(window.location.href)).toBeNull()
    installVisitorAccess()

    expect($visitorAccess.get()).toEqual({ kind: 'open' })
    expect($signInFailed.get()).toBe(false)
  })

  it('leaves the usage-stats offer to the backend answer', () => {
    $sharedMetricsConsent.set(ANSWERED)
    expect($sharedMetricsOfferConsent.get()).toBe($sharedMetricsConsent.get())
    expect($usageStatsOfferPlace.get()).toBeNull()

    $sharedMetricsConsent.set(UNANSWERED)
    expect($sharedMetricsOfferConsent.get()).toBe($sharedMetricsConsent.get())
    expect($usageStatsOfferPlace.get()).toBe('card')
  })
})

describe('the sign-up stand-in', () => {
  it('requires sign up on the next load, with the lab page standing in for the sign-up service', () => {
    window.history.replaceState(null, '', '/?keep=1#/some-chat')
    setDiscoveryLabSignUp(true)

    installVisitorAccess()

    expect($signUpRequired.get()).toBe(true)
    const access = $visitorAccess.get()
    const standIn = new URL(access.kind === 'sign_up_required' ? access.signUpUrl : '')
    expect(standIn.origin).toBe(window.location.origin)
    expect(standIn.searchParams.get('keep')).toBe('1')
    expect(standIn.hash).toBe('#/discovery-lab?stand-in=sign-up')
  })

  it('reads sign up coming back without signing in, once, like the real return', () => {
    setDiscoveryLabSignUp(true)
    window.history.replaceState(null, '', '/?hermes_sign_in=failed#/')

    installVisitorAccess()

    expect($signInFailed.get()).toBe(true)
    // Taken off the address, so a reload does not say it again.
    expect(window.location.search).toBe('')
  })

  it('starts each sign up clean: no earlier outcome or return address rides along', () => {
    setDiscoveryLabSignUp(true)
    const back = `${window.location.origin}/#/`
    const access = discoveryLabVisitorAccess(`${window.location.origin}/?hermes_sign_in=failed&return_to=old#/`)
    const signUpUrl = access?.kind === 'sign_up_required' ? access.signUpUrl : ''

    expect(new URL(signUpUrl).search).toBe('')
    expect(new URL(signUpHref(signUpUrl, back)).searchParams.get('return_to')).toBe(back)
  })

  it('is gone once turned off', () => {
    setDiscoveryLabSignUp(true)
    setDiscoveryLabSignUp(false)

    installVisitorAccess()

    expect($signUpRequired.get()).toBe(false)
  })
})

describe('the usage-stats stand-in', () => {
  it('offers the question where the backend already answered it', () => {
    $sharedMetricsConsent.set(ANSWERED)
    $discoveryLabUsageStatsConsent.set(UNANSWERED)

    expect($usageStatsOfferPlace.get()).toBe('card')

    $guidanceLedger.set({ usageStatsOffer: 'deferred' })
    expect($usageStatsOfferPlace.get()).toBe('tray')
  })

  it('takes an answer without writing the config, and retires', async () => {
    $sharedMetricsConsent.set(ANSWERED)
    $discoveryLabUsageStatsConsent.set(UNANSWERED)
    $sharedMetricsDetailsOpen.set(true)
    const request = vi.fn()

    await answerSharedMetricsOffer(request, 'share', 'gateway|default')

    expect(request).not.toHaveBeenCalled()
    expect($discoveryLabUsageStatsConsent.get()).toBeNull()
    expect($sharedMetricsDetailsOpen.get()).toBe(false)
    expect($usageStatsOfferPlace.get()).toBeNull()
  })
})
