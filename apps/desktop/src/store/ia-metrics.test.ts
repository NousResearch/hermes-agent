import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'

import schema from '../../../../hermes_cli/observability/schemas/hermes.shared_metrics.v4.schema.json'

import {
  bindDesktopMetrics,
  captureDesktopMetricsGate,
  recordAction,
  recordFeatureUse,
  recordFriction,
  recordOnboarding,
  resetDesktopMetricsForTests,
  tickDesktopMetrics
} from './desktop-metrics'
import { immediateMetricsLocks, setMetricsGate, settleMetrics, storedMetricsAggregate } from './desktop-metrics-test-locks'
import { captureIaRecorder, recordIaAction } from './ia-metrics'
import { IA_ACTION_IDS } from './ia-metrics-contract'
import { $shellChoice } from './shell-variant'

// Aggregate writes land after a short coordination phase: settle before reading.
const stored = storedMetricsAggregate

beforeEach(() => {
  immediateMetricsLocks()
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'private-gateway', profile: 'private-profile' })
  bindDesktopMetrics(null, 'private-gateway|private-profile')
  $shellChoice.set(true)
})

afterEach(() => {
  vi.unstubAllGlobals()
  resetDesktopMetricsForTests()
  vi.useRealTimers()
})

it('collects only consenting v2 interactions and closed codes, never identity or arbitrary strings', async () => {
  recordIaAction('ia.rail.capabilities')
  expect(stored()).toBe('')
  await setMetricsGate(captureDesktopMetricsGate, 'on')
  recordIaAction('ia.rail.capabilities')
  recordIaAction('private-session-title' as Parameters<typeof recordIaAction>[0])
  await settleMetrics()
  expect(stored()).toContain('ia.rail.capabilities|click')
  expect(stored()).not.toMatch(/private-/)
  $shellChoice.set(false)
  recordIaAction('ia.rail.chat')
  await settleMetrics()
  expect(stored()).not.toContain('ia.rail.chat')
  await setMetricsGate(captureDesktopMetricsGate, 'off')
  expect(stored()).toBe('')
})

it('drops async results after a scope or consent change, including opt-out then opt-in', async () => {
  await setMetricsGate(captureDesktopMetricsGate, 'on')
  const record = captureIaRecorder()
  record('ia.pin.pinned')
  await settleMetrics()
  expect(stored()).toContain('ia.pin.pinned|state')
  $apiRequestScope.set({ connectionId: 'another', profile: 'private-profile' })
  record('ia.pin.unpinned')
  await settleMetrics()
  expect(stored()).not.toContain('ia.pin.unpinned')
  bindDesktopMetrics(null, 'another|private-profile')
  await setMetricsGate(captureDesktopMetricsGate, 'on')
  const pending = captureIaRecorder()
  await setMetricsGate(captureDesktopMetricsGate, 'off')
  await setMetricsGate(captureDesktopMetricsGate, 'on')
  pending('ia.pin.failed')
  await settleMetrics()
  expect(stored()).not.toContain('ia.pin.failed')
})

it.each([
  { connectionId: 'private-gateway', profile: 'beta' },
  { connectionId: 'beta-gateway', profile: 'private-profile' }
])('isolates recorders and the persisted daily outbox across a switch to $connectionId/$profile', async beta => {
  vi.useFakeTimers()
  vi.setSystemTime(Date.UTC(2026, 8, 27, 12))
  const alpha = { connectionId: 'private-gateway', profile: 'private-profile' }
  const requestA = vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({ recorded: true }) as never)
  const requestB = vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({ recorded: true }) as never)
  const alphaKey = `${alpha.connectionId}|${alpha.profile}`
  const betaKey = `${beta.connectionId}|${beta.profile}`
  const settle = () => vi.advanceTimersByTimeAsync(0)

  bindDesktopMetrics(requestA, alphaKey)
  await setMetricsGate(captureDesktopMetricsGate, 'on')
  const oldRecorder = captureIaRecorder()
  oldRecorder('ia.pin.pinned')
  await settle()
  const alphaState = stored()
  requestA.mockClear()

  $apiRequestScope.set(beta)
  const beforeBind = captureIaRecorder()
  oldRecorder('ia.pin.failed')
  beforeBind('ia.view.chat')
  recordIaAction('ia.rail.chat')
  // The shared sink protects classic actions and immediate events too.
  recordAction('session.new', 'shortcut')
  recordFeatureUse('projects')
  recordFriction('error_toast', 'timeout')
  recordOnboarding('guide', 'reached')
  await setMetricsGate(captureDesktopMetricsGate, 'on') // A late consent answer cannot re-open A's gate.
  bindDesktopMetrics(requestA, alphaKey) // Nor can an obsolete binding.
  beforeBind('ia.view.chat')
  tickDesktopMetrics()
  await settle()
  expect(stored()).toBe(alphaState)
  expect(requestA).not.toHaveBeenCalled()

  bindDesktopMetrics(requestB, betaKey)
  await setMetricsGate(captureDesktopMetricsGate, 'off')
  captureIaRecorder()('ia.view.chat')
  recordIaAction('ia.rail.chat')
  await settle()
  expect(stored()).toBe(alphaState)
  expect(requestB).not.toHaveBeenCalled()

  await setMetricsGate(captureDesktopMetricsGate, 'on')
  beforeBind('ia.view.chat') // Dropped work must not revive after binding/consent.
  oldRecorder('ia.pin.failed')
  const boundRecorder = captureIaRecorder()
  boundRecorder('ia.view.artifacts')
  recordIaAction('ia.rail.capabilities')
  recordAction('session.new', 'shortcut')
  await settle()
  expect(requestB).not.toHaveBeenCalled()

  // Read and drain the real persisted outbox through only mocked RPC requesters.
  bindDesktopMetrics(null, betaKey)
  vi.setSystemTime(Date.UTC(2026, 8, 28, 12))
  tickDesktopMetrics()
  await settle()
  expect(stored()).toContain('"pending":[{')
  bindDesktopMetrics(requestB, betaKey)
  await settle()
  expect(requestB.mock.calls).toEqual([
    ['shared_metrics.desktop_daily', expect.objectContaining({
      actions: [
        { action: 'ia.view.artifacts', count: 1, via: 'state' },
        { action: 'ia.rail.capabilities', count: 1, via: 'click' },
        { action: 'session.new', count: 1, via: 'shortcut' }
      ]
    })]
  ])

  $apiRequestScope.set(alpha)
  bindDesktopMetrics(requestA, alphaKey)
  await setMetricsGate(captureDesktopMetricsGate, 'on')
  await settle()
  expect(requestA.mock.calls).toEqual([
    ['shared_metrics.desktop_daily', expect.objectContaining({
      actions: [{ action: 'ia.pin.pinned', count: 1, via: 'state' }]
    })]
  ])
})

it('keeps every frontend IA counter in the published backend allowlist', () => {
  const definitions = Object.values(schema.$defs) as {
    properties?: { name?: { const: string }; dimensions?: { properties: { action: { enum: string[] } } } }
  }[]

  const actions = definitions.find(definition => definition.properties?.name?.const === 'hermes.desktop.action_use')!
    .properties!.dimensions!.properties.action.enum

  expect(IA_ACTION_IDS.every(action => actions.includes(action))).toBe(true)
})
