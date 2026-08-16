import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

// Registry-agent activation (ensureGatewayAgent — the SDK ensureAgent door).
// Two regressions pinned here:
//  1. Activating an ALREADY-OPEN registry agent must still resync
//     $connection (via getConnectionFor) and move $activeGatewayProfile —
//     previously only a freshly-dialed socket synced $connection (inside
//     openSecondary), so re-activating an open agent left REST/fs/media and
//     image-attach routing on the previous backend (same class as #46651).
//  2. Agent activations share the gatewaySwitch mutex with profile switches —
//     without it, two rapid activations could complete out of order and the
//     EARLIER setActive() landed last.

const ensureGatewayForAgent = vi.fn(async (_connectionId: null | string, _profile: string) => undefined)
const ensureGatewayForProfile = vi.fn(async (_profile: string) => undefined)
const openGatewayForProfile = vi.fn(async (_profile: string) => undefined)
const $gateway = atom<unknown>({ id: 'live-socket' })
const resetStarmapGraph = vi.fn()

vi.mock('@/store/gateway', () => ({
  $gateway,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  openGatewayForProfile
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph }))

const { $activeGatewayProfile, ensureGatewayAgent, ensureGatewayProfile } = await import('./profile')
const { $connection } = await import('./session')

const agentConn = (over: Partial<HermesConnection> = {}): HermesConnection =>
  ({ baseUrl: 'https://homelab.invalid', mode: 'remote', profile: 'research', ...over }) as HermesConnection

const localConn = (over: Partial<HermesConnection> = {}): HermesConnection =>
  ({ baseUrl: '', mode: 'local', profile: 'default', ...over }) as HermesConnection

const getConnection = vi.fn<(profile?: string | null) => Promise<HermesConnection>>()

const getConnectionFor =
  vi.fn<(payload: { connectionId?: null | string; profile?: null | string }) => Promise<HermesConnection>>()

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void

  const promise = new Promise<void>(r => {
    resolve = r
  })

  return { promise, resolve }
}

beforeEach(() => {
  getConnection.mockReset()
  getConnectionFor.mockReset()
  ensureGatewayForAgent.mockClear()
  ensureGatewayForProfile.mockClear()
  $gateway.set({ id: 'live-socket' })
  $activeGatewayProfile.set('default')
  $connection.set(localConn())
  vi.stubGlobal('window', { hermesDesktop: { getConnection, getConnectionFor } })
})

afterEach(() => {
  vi.unstubAllGlobals()
  $connection.set(null)
})

describe('ensureGatewayAgent → $connection / $activeGatewayProfile sync', () => {
  it('resyncs $connection and $activeGatewayProfile even when the agent socket is already open', async () => {
    // The store-level activation resolves instantly (socket already open) —
    // exactly the case that used to skip the sync entirely.
    getConnectionFor.mockResolvedValue(agentConn())

    await ensureGatewayAgent('homelab', 'research')

    expect(ensureGatewayForAgent).toHaveBeenCalledWith('homelab', 'research')
    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'homelab', profile: 'research' })
    expect($activeGatewayProfile.get()).toBe('research')
    expect($connection.get()?.mode).toBe('remote')
    expect($connection.get()?.profile).toBe('research')
  })

  it('leaves the prior connection intact when the descriptor fetch fails', async () => {
    getConnectionFor.mockRejectedValue(new Error('source unreachable'))

    await ensureGatewayAgent('homelab', 'research')

    expect($activeGatewayProfile.get()).toBe('research')
    // Best-effort: boot/reconnect resyncs later; we must not null it out here.
    expect($connection.get()?.mode).toBe('local')
  })

  it('falls through to the profile path for a local/null connectionId', async () => {
    getConnection.mockResolvedValue(agentConn({ mode: 'local', profile: 'research' }))

    await ensureGatewayAgent(null, 'research')

    expect(ensureGatewayForProfile).toHaveBeenCalledWith('research')
    expect(ensureGatewayForAgent).not.toHaveBeenCalled()
    expect(getConnectionFor).not.toHaveBeenCalled()
  })
})

describe('ensureGatewayAgent shares the gatewaySwitch mutex with profile switches', () => {
  it('serializes an agent activation behind an in-flight profile switch', async () => {
    const profileGate = deferred()
    const order: string[] = []

    ensureGatewayForProfile.mockImplementation(async (profile: string) => {
      order.push(`profile:${profile}`)
      await profileGate.promise
    })
    ensureGatewayForAgent.mockImplementation(async (_connectionId, profile) => {
      order.push(`agent:${profile}`)
    })
    getConnection.mockResolvedValue(localConn({ profile: 'worker' }))
    getConnectionFor.mockResolvedValue(agentConn())

    // Start a profile switch that stalls mid-flight, then an agent
    // activation. The agent activation must NOT start until the profile
    // switch settles — otherwise the earlier setActive could land last.
    const profileSwitch = ensureGatewayProfile('worker')
    await Promise.resolve()
    const agentSwitch = ensureGatewayAgent('homelab', 'research')
    await Promise.resolve()

    expect(order).toEqual(['profile:worker'])

    profileGate.resolve()
    await profileSwitch
    await agentSwitch

    expect(order).toEqual(['profile:worker', 'agent:research'])
    // The LAST activation wins the active pointer.
    expect($activeGatewayProfile.get()).toBe('research')
    expect($connection.get()?.profile).toBe('research')
  })

  it('serializes a profile switch behind an in-flight agent activation', async () => {
    const agentGate = deferred()
    const order: string[] = []

    ensureGatewayForAgent.mockImplementation(async (_connectionId, profile) => {
      order.push(`agent:${profile}`)
      await agentGate.promise
    })
    ensureGatewayForProfile.mockImplementation(async (profile: string) => {
      order.push(`profile:${profile}`)
    })
    getConnection.mockResolvedValue(localConn({ profile: 'worker' }))
    getConnectionFor.mockResolvedValue(agentConn())

    const agentSwitch = ensureGatewayAgent('homelab', 'research')
    await Promise.resolve()
    const profileSwitch = ensureGatewayProfile('worker')
    await Promise.resolve()

    expect(order).toEqual(['agent:research'])

    agentGate.resolve()
    await agentSwitch
    await profileSwitch

    expect(order).toEqual(['agent:research', 'profile:worker'])
    expect($activeGatewayProfile.get()).toBe('worker')
  })
})
