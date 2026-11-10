import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopAgentRoster } from '@/global'

import { $fleetRoster, _resetFleetRosterForTests, refreshFleetRoster } from './fleet-roster'

afterEach(() => {
  _resetFleetRosterForTests()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

const unreachable: DesktopAgentRoster = {
  agents: [],
  sources: [{ connectionId: 'lab', label: 'Lab', kind: 'remote', reachable: false, error: 'timed out' }]
}

const recovered: DesktopAgentRoster = {
  agents: [
    { connectionId: 'lab', connectionLabel: 'Lab', connectionKind: 'remote', profile: 'default', handle: 'default' }
  ],
  sources: [{ connectionId: 'lab', label: 'Lab', kind: 'remote', reachable: true }]
}

describe('fleet roster recovery', () => {
  it('queues one fresh enumeration when recovery overlaps an older request', async () => {
    let finish!: (roster: DesktopAgentRoster) => void

    const pending = new Promise<DesktopAgentRoster>(resolve => {
      finish = resolve
    })

    const getAgentRoster = vi.fn().mockReturnValueOnce(pending).mockResolvedValue(recovered)
    vi.stubGlobal('window', { hermesDesktop: { getAgentRoster } })

    const initial = refreshFleetRoster()
    const recovery = refreshFleetRoster({ force: true })
    const repeated = refreshFleetRoster({ force: true })
    expect(getAgentRoster).toHaveBeenCalledTimes(1)
    finish(unreachable)
    await Promise.all([initial, recovery, repeated])
    expect(getAgentRoster).toHaveBeenCalledTimes(2)
    expect($fleetRoster.get()).toBe(recovered)
  })
  it('keeps on-demand sources cached but lets explicit recovery bypass the cooldown', async () => {
    vi.useFakeTimers()

    const onDemand: DesktopAgentRoster = {
      agents: [],
      sources: [{ connectionId: 'ssh', label: 'Box', kind: 'ssh', reachable: false, error: 'connect-on-demand' }]
    }

    const getAgentRoster = vi.fn().mockResolvedValueOnce(onDemand).mockResolvedValue(recovered)
    vi.stubGlobal('window', { hermesDesktop: { getAgentRoster } })

    await refreshFleetRoster()
    await vi.advanceTimersByTimeAsync(5_000)
    await refreshFleetRoster()
    expect(getAgentRoster).toHaveBeenCalledTimes(1)

    await refreshFleetRoster({ force: true })
    expect(getAgentRoster).toHaveBeenCalledTimes(2)
    expect($fleetRoster.get()).toBe(recovered)
  })

  it('retries incomplete rosters before the normal stale window without a focus-event hot loop', async () => {
    vi.useFakeTimers()
    const getAgentRoster = vi.fn().mockResolvedValueOnce(unreachable).mockResolvedValue(recovered)
    vi.stubGlobal('window', { hermesDesktop: { getAgentRoster } })

    await refreshFleetRoster()
    await refreshFleetRoster()
    expect(getAgentRoster).toHaveBeenCalledTimes(1)
    expect($fleetRoster.get()).toBe(unreachable)

    await vi.advanceTimersByTimeAsync(5_000)
    await Promise.all([refreshFleetRoster(), refreshFleetRoster()])
    expect(getAgentRoster).toHaveBeenCalledTimes(2)
    expect($fleetRoster.get()).toBe(recovered)

    await vi.advanceTimersByTimeAsync(5_000)
    await refreshFleetRoster()
    expect(getAgentRoster).toHaveBeenCalledTimes(2)
  })
})
