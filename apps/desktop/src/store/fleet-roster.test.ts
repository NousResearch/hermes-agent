import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $fleetRoster,
  _resetFleetRosterForTests,
  fleetRosterIsComplete,
  refreshFleetRoster
} from './fleet-roster'

describe('fleetRosterIsComplete (#101339)', () => {
  it('treats missing or empty sources as complete (nothing to retry)', () => {
    expect(fleetRosterIsComplete(null)).toBe(true)
    expect(fleetRosterIsComplete({ agents: [], sources: [] } as never)).toBe(true)
  })

  it('is incomplete when a registered source is unreachable', () => {
    expect(
      fleetRosterIsComplete({
        agents: [],
        sources: [
          { connectionId: 'local', kind: 'local', label: 'This device', reachable: true },
          { connectionId: 'gw', kind: 'remote', label: 'Lab', reachable: false, error: 'roster enumeration timed out' }
        ]
      } as never)
    ).toBe(false)
  })

  it('treats connect-on-demand as complete (deliberate skip, not a failed dial)', () => {
    expect(
      fleetRosterIsComplete({
        agents: [],
        sources: [{ connectionId: 'ssh-1', kind: 'ssh', label: 'Box', reachable: false, error: 'connect-on-demand' }]
      } as never)
    ).toBe(true)
  })
})

describe('refreshFleetRoster incomplete-cache bypass (#101339)', () => {
  afterEach(() => {
    _resetFleetRosterForTests()
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('refetches within the stale window when the cached roster has an unreachable remote', async () => {
    const getAgentRoster = vi
      .fn()
      .mockResolvedValueOnce({
        agents: [],
        sources: [
          { connectionId: 'local', kind: 'local', label: 'This device', reachable: true },
          { connectionId: 'gw', kind: 'remote', label: 'Lab', reachable: false, error: 'timed out' }
        ]
      })
      .mockResolvedValueOnce({
        agents: [{ connectionId: 'gw', profile: 'default', handle: 'default' }],
        sources: [
          { connectionId: 'local', kind: 'local', label: 'This device', reachable: true },
          { connectionId: 'gw', kind: 'remote', label: 'Lab', reachable: true }
        ]
      })

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { getAgentRoster }

    await refreshFleetRoster()
    expect(getAgentRoster).toHaveBeenCalledTimes(1)
    expect($fleetRoster.get()?.sources?.[1]?.reachable).toBe(false)

    await refreshFleetRoster()
    expect(getAgentRoster).toHaveBeenCalledTimes(2)
    expect($fleetRoster.get()?.sources?.[1]?.reachable).toBe(true)
  })

  it('still skips refetch within the stale window for a complete roster', async () => {
    const getAgentRoster = vi.fn().mockResolvedValue({
      agents: [],
      sources: [{ connectionId: 'local', kind: 'local', label: 'This device', reachable: true }]
    })

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { getAgentRoster }

    await refreshFleetRoster()
    await refreshFleetRoster()
    expect(getAgentRoster).toHaveBeenCalledTimes(1)
  })
})
