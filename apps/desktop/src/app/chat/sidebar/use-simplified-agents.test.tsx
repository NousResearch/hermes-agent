import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $fleetRoster } from '@/store/fleet-roster'
import { $profiles } from '@/store/profile'

import { useSimplifiedAgents } from './use-simplified-agents'

const roster = (sources: { connectionId: string; label: string }[]) =>
  ({
    primaryConnectionId: 'local',
    agents: [],
    sources: sources.map(source => ({ ...source, kind: 'local', reachable: true }))
  }) as unknown as NonNullable<typeof $fleetRoster.value>

beforeEach(() => {
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
  $apiRequestScope.set({ connectionId: 'local', profile: 'default' })
  $connectionsRegistry.set(null)
  $fleetRoster.set(null)
  $profiles.set([{ name: 'default', is_default: true }] as typeof $profiles.value)
})

afterEach(() => {
  delete window.document.documentElement.dataset.hermesDesktopHost
  $fleetRoster.set(null)
  $apiRequestScope.set({ connectionId: null, profile: null })
})

it('keeps the serving machine label stable while the roster loads its raw address', () => {
  const { result } = renderHook(() => useSimplifiedAgents([]))
  const before = result.current.agents[0].gateway

  act(() => $fleetRoster.set(roster([{ connectionId: 'local', label: window.location.host }])))

  expect(result.current.agents[0].gateway).toBe(before)
  expect(before).not.toBe(window.location.host)
})

it('keeps a roster label that names the machine rather than its address', () => {
  const { result } = renderHook(() => useSimplifiedAgents([]))

  act(() => $fleetRoster.set(roster([{ connectionId: 'local', label: 'Studio Mac' }])))

  expect(result.current.agents[0].gateway).toBe('Studio Mac')
})
