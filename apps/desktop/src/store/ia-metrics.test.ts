import { beforeEach, expect, it } from 'vitest'

import { $apiRequestScope } from '@/api/client'

import schema from '../../../../hermes_cli/observability/schemas/hermes.shared_metrics.v4.schema.json'

import { DESKTOP_METRICS_STATE_KEY, resetDesktopMetricsForTests, setDesktopMetricsGate } from './desktop-metrics'
import { captureIaRecorder, recordIaAction } from './ia-metrics'
import { IA_ACTION_IDS } from './ia-metrics-contract'
import { $agentSidebarChoice } from './simplified-agent-mode'

function stored(): string {
  return Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '')
    .filter(key => key.startsWith(DESKTOP_METRICS_STATE_KEY))
    .map(key => window.localStorage.getItem(key))
    .join('')
}

beforeEach(() => {
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'private-gateway', profile: 'private-profile' })
  $agentSidebarChoice.set(true)
})

it('collects only consenting v2 interactions and closed codes, never identity or arbitrary strings', () => {
  recordIaAction('ia.rail.capabilities')
  expect(stored()).toBe('')
  setDesktopMetricsGate('on')
  recordIaAction('ia.rail.capabilities')
  recordIaAction('private-session-title' as Parameters<typeof recordIaAction>[0])
  expect(stored()).toContain('ia.rail.capabilities|click')
  expect(stored()).not.toMatch(/private-/)
  $agentSidebarChoice.set(false)
  recordIaAction('ia.rail.chat')
  expect(stored()).not.toContain('ia.rail.chat')
  setDesktopMetricsGate('off')
  expect(stored()).toBe('')
})

it('drops async results after a scope or consent change, including opt-out then opt-in', () => {
  setDesktopMetricsGate('on')
  const record = captureIaRecorder()
  record('ia.pin.pinned')
  expect(stored()).toContain('ia.pin.pinned|state')
  $apiRequestScope.set({ connectionId: 'another', profile: 'private-profile' })
  record('ia.pin.unpinned')
  expect(stored()).not.toContain('ia.pin.unpinned')
  const pending = captureIaRecorder()
  setDesktopMetricsGate('off')
  setDesktopMetricsGate('on')
  pending('ia.pin.failed')
  expect(stored()).not.toContain('ia.pin.failed')
})

it('keeps every frontend IA counter in the published backend allowlist', () => {
  const definitions = Object.values(schema.$defs) as {
    properties?: { name?: { const: string }; dimensions?: { properties: { action: { enum: string[] } } } }
  }[]

  const actions = definitions.find(definition => definition.properties?.name?.const === 'hermes.desktop.action_use')!
    .properties!.dimensions!.properties.action.enum

  expect(IA_ACTION_IDS.every(action => actions.includes(action))).toBe(true)
})
