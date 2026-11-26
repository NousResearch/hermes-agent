import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'

import {
  bindDesktopMetrics,
  captureDesktopMetricsGate,
  DESKTOP_METRICS_STATE_KEY,
  resetDesktopMetricsForTests
} from './desktop-metrics'
import { immediateMetricsLocks, setMetricsGate } from './desktop-metrics-test-locks'
import {
  $firstTimeSplash,
  $firstTimeSplashPlace,
  $firstTimeSplashReopenable,
  $newcomer,
  clearFirstTimeSplashPlace,
  closeFirstTimeSplash,
  considerFirstTimeSplash,
  FIRST_TIME_PLACES,
  noteFirstTimeSplashRow,
  reopenFirstTimeSplash
} from './first-time-splash'
import { $guidanceLedger } from './guidance-ledger'
import { $renameSplash, considerRenameSplash } from './rename-splash'
import { $shellChoice } from './shell-variant'

function recorded(): string {
  return Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '')
    .filter(key => key.startsWith(DESKTOP_METRICS_STATE_KEY))
    .map(key => window.localStorage.getItem(key))
    .join('')
}

/** What the shell does when the verdict settles: both splashes consider it, rename first. */
function settle(verdict: 'existing' | 'fresh' | 'unknown') {
  considerRenameSplash(verdict)
  considerFirstTimeSplash(verdict)
}

beforeEach(() => {
  immediateMetricsLocks()
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'gateway', profile: 'default' })
  bindDesktopMetrics(null, 'gateway|default')
  $shellChoice.set(true)
  $guidanceLedger.set({})
  $renameSplash.set(null)
  $firstTimeSplash.set(null)
  $firstTimeSplashPlace.set(null)
})

afterEach(() => {
  resetDesktopMetricsForTests()
  vi.unstubAllGlobals()
})

describe('who gets which splash', () => {
  it('a new install gets the first-time splash, never the rename splash', () => {
    settle('fresh')

    expect($firstTimeSplash.get()).toBe('shown')
    expect($renameSplash.get()).toBeNull()
    expect($newcomer.get()).toBe(true)
  })

  it('someone who used Hermes before gets the rename splash, never the first-time splash', () => {
    settle('existing')

    expect($renameSplash.get()).toBe('shown')
    expect($firstTimeSplash.get()).toBeNull()
    expect($guidanceLedger.get().firstTimeSplash).toBe('not_applicable')
  })

  it('nobody gets either while the machine has not answered', () => {
    settle('unknown')

    expect($renameSplash.get()).toBeNull()
    expect($firstTimeSplash.get()).toBeNull()
    expect($guidanceLedger.get()).toEqual({})
  })

  it('a newcomer who left without closing it still gets it after their first chat exists', () => {
    settle('fresh')
    $firstTimeSplash.set(null) // the page unloads; nothing was acknowledged

    settle('existing')
    expect($firstTimeSplash.get()).toBe('shown')
    expect($renameSplash.get()).toBeNull()
  })
})

describe('the first-time splash', () => {
  it('opens at rest: the four rail places, no row lit, and nothing left lit after a close', () => {
    expect(FIRST_TIME_PLACES).toEqual(['chats', 'agents', 'connections', 'routines'])

    settle('fresh')
    expect($firstTimeSplash.get()).toBe('shown')
    expect($firstTimeSplashPlace.get()).toBeNull()

    noteFirstTimeSplashRow('chats')
    expect($firstTimeSplashPlace.get()).toBe('chats')

    closeFirstTimeSplash('close')
    expect($firstTimeSplashPlace.get()).toBeNull()

    reopenFirstTimeSplash()
    expect($firstTimeSplash.get()).toBe('reopened')
    expect($firstTimeSplashPlace.get()).toBeNull()
  })

  it('lights the row reached and clears back to rest when it steps away', () => {
    settle('fresh')

    noteFirstTimeSplashRow('connections')
    expect($firstTimeSplashPlace.get()).toBe('connections')

    clearFirstTimeSplashPlace()
    expect($firstTimeSplashPlace.get()).toBeNull()

    // While the splash is closed there is no row to light.
    closeFirstTimeSplash('escape')
    noteFirstTimeSplashRow('routines')
    expect($firstTimeSplashPlace.get()).toBeNull()
  })

  it('shows once: any close marks it seen, and the next visit does not show it', () => {
    for (const how of ['close', 'escape', 'nav', 'signup'] as const) {
      $guidanceLedger.set({})
      settle('fresh')
      closeFirstTimeSplash(how)

      expect($firstTimeSplash.get()).toBeNull()
      expect($guidanceLedger.get().firstTimeSplash).toBe('seen')

      settle('fresh')
      expect($firstTimeSplash.get()).toBeNull()
    }
  })

  it('reopens from the tray only once seen', () => {
    reopenFirstTimeSplash()
    expect($firstTimeSplash.get()).toBeNull()

    settle('fresh')
    closeFirstTimeSplash('close')
    expect($firstTimeSplashReopenable.get()).toBe(true)

    reopenFirstTimeSplash()
    expect($firstTimeSplash.get()).toBe('reopened')
  })

  it('records closed codes only behind consent: shown, each row once per showing, how it closed', async () => {
    settle('fresh')
    closeFirstTimeSplash('nav')
    expect(recorded()).toBe('')

    await setMetricsGate(captureDesktopMetricsGate, 'on')
    reopenFirstTimeSplash()
    noteFirstTimeSplashRow('theme')
    noteFirstTimeSplashRow('theme')
    noteFirstTimeSplashRow('routines')
    noteFirstTimeSplashRow('connections')
    closeFirstTimeSplash('escape')

    const state = recorded()
    expect(state).toContain('ia.splash.first_time_reopened|state')
    expect(state).toContain('"ia.splash_row.first_time_theme|state":1')
    expect(state).toContain('ia.splash_row.first_time_routines|state')
    expect(state).toContain('ia.splash_row.first_time_connections|state')
    expect(state).toContain('ia.splash.first_time_closed_escape|shortcut')
    expect(state).not.toContain('"other')
  })
})
