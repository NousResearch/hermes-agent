import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import type { SidebarSessionsResponse } from '@/api/sessions'
import { makeSessionInfo } from '@/test/session-info'

import {
  bindDesktopMetrics,
  captureDesktopMetricsGate,
  resetDesktopMetricsForTests
} from './desktop-metrics'
import { immediateMetricsLocks, setMetricsGate, settleMetrics, storedMetricsAggregate } from './desktop-metrics-test-locks'
import { captureIaPinCounts } from './ia-pin-metrics'
import { $shellChoice } from './shell-variant'

const row = (id: string, profile = 'default', pinned = true) =>
  makeSessionInfo({ id, profile, pinned, connection_id: 'alpha' })

const page = (sessions = [] as ReturnType<typeof row>[]): SidebarSessionsResponse => ({
  recents: { sessions },
  cron: { sessions: [] },
  messaging: { sessions: [] }
})

// Aggregate writes land after a short coordination phase.
const actions = async () => {
  await settleMetrics()
  const stored = storedMetricsAggregate()

  return stored ? JSON.parse(stored).today.actions : {}
}

beforeEach(async () => {
  immediateMetricsLocks()
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $shellChoice.set(true)
  $apiRequestScope.set({ connectionId: 'alpha', profile: 'default' })
  bindDesktopMetrics(null, 'alpha|default')
  await setMetricsGate(captureDesktopMetricsGate, 'on')
})
afterEach(() => {
  vi.unstubAllGlobals()
  resetDesktopMetricsForTests()
  $shellChoice.set(false)
  $apiRequestScope.set({ connectionId: null, profile: null })
})

it('counts backfilled pins beyond the recency window, deduplicates slices and separates the consenting owner', async () => {
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
  expect((await actions())['ia.pin_count.overflow|state']).toBe(1)
  expect((await actions())['ia.profile_pin_count.three|state']).toBe(1)
  expect(JSON.stringify(await actions())).not.toMatch(/private-agent|alpha|beta|old-pin/)
})

it('reports confirmed empty lists but does not fabricate totals from failed or legacy reads', async () => {
  captureIaPinCounts('all')({ ...page(), recents: { sessions: [], failed: true } })
  captureIaPinCounts('all')({ ...page(), errors: [{ profile: 'default', error: 'offline' }] })
  captureIaPinCounts('all')(page([{ ...row('legacy'), pinned: undefined }]))
  expect(await actions()).toEqual({})
  captureIaPinCounts('all')(page())
  expect((await actions())['ia.pin_count.zero|state']).toBe(1)
  expect((await actions())['ia.profile_pin_count.zero|state']).toBe(1)
})

it('keeps a successful profile sample when another profile fails, but skips the total and mismatched scope', async () => {
  captureIaPinCounts('all')({ ...page([row('pin')]), profiles_failed: { research: { failed: true } } })
  expect(await actions()).toEqual({ 'ia.profile_pin_count.one|state': 1 })
  captureIaPinCounts('research')(page([row('pin', 'research')]))
  expect((await actions())['ia.profile_pin_count.zero|state']).toBeUndefined()
})

it('does not collect before consent, after revocation, across a gateway switch, or with v2 disabled', async () => {
  await setMetricsGate(captureDesktopMetricsGate, 'off')
  const beforeConsent = captureIaPinCounts('all')
  await setMetricsGate(captureDesktopMetricsGate, 'on')
  beforeConsent(page())
  const pending = captureIaPinCounts('all')
  await setMetricsGate(captureDesktopMetricsGate, 'off')
  await setMetricsGate(captureDesktopMetricsGate, 'on')
  pending(page())
  const oldOwner = captureIaPinCounts('all')
  $apiRequestScope.set({ connectionId: 'beta', profile: 'default' })
  bindDesktopMetrics(null, 'beta|default')
  await setMetricsGate(captureDesktopMetricsGate, 'on')
  oldOwner(page())
  $shellChoice.set(false)
  captureIaPinCounts('all')(page())
  expect(await actions()).toEqual({})
})
