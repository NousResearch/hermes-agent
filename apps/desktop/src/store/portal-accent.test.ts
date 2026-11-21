import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { isSharedPreferenceKey } from '@/lib/ui-preferences/keys'

import {
  bindDesktopMetrics,
  captureDesktopMetricsGate,
  DESKTOP_METRICS_STATE_KEY,
  resetDesktopMetricsForTests
} from './desktop-metrics'
import { immediateMetricsLocks, setMetricsGate } from './desktop-metrics-test-locks'
import { pickPortalAccent, PORTAL_ACCENTS } from './portal-accent'
import { $shellChoice } from './shell-variant'

function recorded(): string {
  return Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '')
    .filter(key => key.startsWith(DESKTOP_METRICS_STATE_KEY))
    .map(key => window.localStorage.getItem(key))
    .join('')
}

/** One page load: the store reads what this browser kept. */
async function load() {
  vi.resetModules()

  return import('./portal-accent')
}

beforeEach(() => {
  immediateMetricsLocks()
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'gateway', profile: 'default' })
  bindDesktopMetrics(null, 'gateway|default')
  $shellChoice.set(true)
})

afterEach(() => {
  vi.unstubAllGlobals()
  delete document.documentElement.dataset.portalAccent
  resetDesktopMetricsForTests()
})

describe('the redesign’s accent', () => {
  it('starts on the blue, paints a pick on <html> at once, and keeps it for the next visit', async () => {
    const first = await load()
    const stop = first.installPortalAccentAttribute()
    expect(document.documentElement.dataset.portalAccent).toBeUndefined()

    first.pickPortalAccent('violet')
    expect(document.documentElement.dataset.portalAccent).toBe('violet')
    stop()

    const second = await load()
    second.installPortalAccentAttribute()
    expect(second.$portalAccent.get()).toBe('violet')

    second.pickPortalAccent('blue')
    expect(document.documentElement.dataset.portalAccent).toBeUndefined()
  })

  it('is saved where the light / dark mode is: a machine-level preference that moves with the account', async () => {
    const { PORTAL_ACCENT_KEY } = await load()

    expect(isSharedPreferenceKey(PORTAL_ACCENT_KEY)).toBe(true)
    expect(isSharedPreferenceKey('hermes-desktop-mode-v1')).toBe(true)
  })

  it('reads a damaged stored value as the blue', async () => {
    window.localStorage.setItem('hermes.desktop.portalAccent.v1', 'chartreuse')

    expect((await load()).$portalAccent.get()).toBe('blue')
  })

  it('counts a pick by its colour’s code, behind consent', async () => {
    pickPortalAccent('pink')
    expect(recorded()).toBe('')

    await setMetricsGate(captureDesktopMetricsGate, 'on')
    pickPortalAccent('orange')
    pickPortalAccent('orange')

    expect(recorded()).toContain('"ia.theme_pick.color_orange|click":1')
  })

  it('counts every one of the eight colours, the bright ones included', async () => {
    await setMetricsGate(captureDesktopMetricsGate, 'on')

    for (const accent of PORTAL_ACCENTS.filter(accent => accent !== 'blue')) {
      pickPortalAccent(accent)
      expect(recorded()).toContain(`"ia.theme_pick.color_${accent}|click":1`)
    }
  })
})
