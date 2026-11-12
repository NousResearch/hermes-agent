import { cleanup, renderHook } from '@testing-library/react'
import { act } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import type { DesktopAgentRoster } from '@/global'
import { _resetFleetRosterForTests } from '@/store/fleet-roster'
import { $gatewayState } from '@/store/session'

import { useFleetRoster } from './use-fleet-roster'

afterEach(() => {
  cleanup()
  _resetFleetRosterForTests()
  $gatewayState.set('idle')
  vi.unstubAllGlobals()
})

it('re-enumerates a roster that timed out while the local gateway was still booting', async () => {
  // A cold `gateway ensure` outlives the roster's per-source deadline, so the
  // first enumeration reports This device unreachable. Nothing else refetches
  // it until a window focus, and the rail counted This device as one square.
  const booting: DesktopAgentRoster = {
    agents: [],
    sources: [{ connectionId: 'local', label: 'This device', kind: 'local', reachable: false, error: 'timed out' }]
  }

  const getAgentRoster = vi.fn().mockResolvedValue(booting)
  window.hermesDesktop = { getAgentRoster } as never

  $gatewayState.set('connecting')
  renderHook(() => useFleetRoster(true))
  await act(async () => undefined)
  expect(getAgentRoster).toHaveBeenCalledTimes(1)

  await act(async () => {
    $gatewayState.set('open')
  })

  expect(getAgentRoster).toHaveBeenCalledTimes(2)
})
