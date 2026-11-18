import { afterEach, beforeEach, expect, it } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import type { SidebarSessionsResponse } from '@/api/sessions'
import { makeSessionInfo } from '@/test/session-info'

import {
  bindDesktopMetrics,
  DESKTOP_METRICS_STATE_KEY,
  resetDesktopMetricsForTests,
  setDesktopMetricsGate
} from './desktop-metrics'
import { captureIaPinCounts } from './ia-pin-metrics'
import { $shellChoice } from './shell-variant'

const row = (id: string, profile = 'default', pinned = true) =>
  makeSessionInfo({ id, profile, pinned, connection_id: 'alpha' })

const page = (sessions = [] as ReturnType<typeof row>[]): SidebarSessionsResponse => ({
  recents: { sessions },
  cron: { sessions: [] },
  messaging: { sessions: [] }
})

const actions = () => {
  const key = Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '').find(key =>
    key.startsWith(DESKTOP_METRICS_STATE_KEY)
  )

  return key ? JSON.parse(window.localStorage.getItem(key)!).today.actions : {}
}

beforeEach(() => {
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $shellChoice.set(true)
  $apiRequestScope.set({ connectionId: 'alpha', profile: 'default' })
  bindDesktopMetrics(null, 'alpha|default')
  setDesktopMetricsGate('on')
})
afterEach(() => {
  resetDesktopMetricsForTests()
  $shellChoice.set(false)
  $apiRequestScope.set({ connectionId: null, profile: null })
})

it('counts backfilled pins beyond the recency window, deduplicates slices and separates the consenting owner', () => {
  const snapshot = page([
    ...Array.from({ length: 50 }, (_, i) => row(`recent-${i}`, 'default', false)),
    row('old-pin'),
    row('second'),
    row('third'),
    row('other-profile', 'private-agent'),
    { ...row('remote'), connection_id: 'beta' },
    { ...row('hidden'), hidden: true },
    { ...row('archived'), archived: true }
  ])

  snapshot.cron.sessions = [row('old-pin')]
  captureIaPinCounts('all')(snapshot)
  expect(actions()['ia.pin_count.overflow|state']).toBe(1)
  expect(actions()['ia.profile_pin_count.three|state']).toBe(1)
  expect(JSON.stringify(actions())).not.toMatch(/private-agent|alpha|beta|old-pin/)
})

it('reports confirmed empty lists but does not fabricate totals from failed or legacy reads', () => {
  captureIaPinCounts('all')({ ...page(), recents: { sessions: [], failed: true } })
  captureIaPinCounts('all')({ ...page(), errors: [{ profile: 'default', error: 'offline' }] })
  captureIaPinCounts('all')(page([{ ...row('legacy'), pinned: undefined }]))
  expect(actions()).toEqual({})
  captureIaPinCounts('all')(page())
  expect(actions()['ia.pin_count.zero|state']).toBe(1)
  expect(actions()['ia.profile_pin_count.zero|state']).toBe(1)
})

it('keeps a successful profile sample when another profile fails, but skips the total and mismatched scope', () => {
  captureIaPinCounts('all')({ ...page([row('pin')]), profiles_failed: { research: { failed: true } } })
  expect(actions()).toEqual({ 'ia.profile_pin_count.one|state': 1 })
  captureIaPinCounts('research')(page([row('pin', 'research')]))
  expect(actions()['ia.profile_pin_count.zero|state']).toBeUndefined()
})

it('does not collect before consent, after revocation, across a gateway switch, or with v2 disabled', () => {
  setDesktopMetricsGate('off')
  const beforeConsent = captureIaPinCounts('all')
  setDesktopMetricsGate('on')
  beforeConsent(page())
  const pending = captureIaPinCounts('all')
  setDesktopMetricsGate('off')
  setDesktopMetricsGate('on')
  pending(page())
  const oldOwner = captureIaPinCounts('all')
  $apiRequestScope.set({ connectionId: 'beta', profile: 'default' })
  bindDesktopMetrics(null, 'beta|default')
  setDesktopMetricsGate('on')
  oldOwner(page())
  $shellChoice.set(false)
  captureIaPinCounts('all')(page())
  expect(actions()).toEqual({})
})
