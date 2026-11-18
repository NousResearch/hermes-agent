import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'

import {
  bindDesktopMetrics,
  DESKTOP_METRICS_STATE_KEY,
  resetDesktopMetricsForTests,
  setDesktopMetricsGate
} from './desktop-metrics'
import { $shellChoice } from './shell-variant'
import { $signInFailed, $signUpRequired, $visitorAccess, installVisitorAccess, startSignUp } from './visitor-access'

type HostWindow = Window & { __HERMES_SIGN_UP_URL__?: string }

function recorded(): string {
  return Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '')
    .filter(key => key.startsWith(DESKTOP_METRICS_STATE_KEY))
    .map(key => window.localStorage.getItem(key))
    .join('')
}

beforeEach(() => {
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'gateway', profile: 'default' })
  bindDesktopMetrics(null, 'gateway|default')
  $shellChoice.set(true)
  $visitorAccess.set({ kind: 'open' })
  $signInFailed.set(false)
})

afterEach(() => {
  delete (window as HostWindow).__HERMES_SIGN_UP_URL__
  window.history.replaceState(null, '', '/')
  vi.unstubAllGlobals()
  resetDesktopMetricsForTests()
})

describe('visitor access on this page', () => {
  it('stays open where the serving page names no sign up (every page served today)', () => {
    installVisitorAccess()

    expect($signUpRequired.get()).toBe(false)
  })

  it('requires sign up where the serving page names it, and reads a failed sign in once', () => {
    ;(window as HostWindow).__HERMES_SIGN_UP_URL__ = 'https://portal.example/sign-up'
    window.history.replaceState(null, '', '/?hermes_sign_in=failed#/')

    installVisitorAccess()

    expect($signUpRequired.get()).toBe(true)
    expect($signInFailed.get()).toBe(true)
    // Taken off the address, so a reload does not say it again.
    expect(window.location.search).toBe('')
  })

  it('leaves for sign up, to come back here, and counts where it started', () => {
    const assign = vi.fn()
    vi.stubGlobal('location', { href: 'https://hermes.example/#/', assign })
    $visitorAccess.set({ kind: 'sign_up_required', signUpUrl: 'https://portal.example/sign-up' })
    $signInFailed.set(true)
    setDesktopMetricsGate('on')

    startSignUp('chat')

    expect(assign).toHaveBeenCalledWith(
      `https://portal.example/sign-up?return_to=${encodeURIComponent('https://hermes.example/#/')}`
    )
    expect($signInFailed.get()).toBe(false)
    expect(recorded()).toContain('ia.sign_up.chat|click')
  })

  it('never leaves for sign up where it is not required', () => {
    const assign = vi.fn()
    vi.stubGlobal('location', { href: 'https://hermes.example/#/', assign })

    startSignUp('splash')

    expect(assign).not.toHaveBeenCalled()
  })
})
