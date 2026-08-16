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
//  3. A SUCCEEDING activation publishes the gateway, the profile pointer and
//     the connection descriptor with no asynchronous gap between them.
//     Activating first and awaiting the descriptor after left $gateway on the
//     new backend while $connection still described the old one.
//  4. A FAILING descriptor lookup publishes none of the three. Swallowing the
//     rejection and publishing anyway produced the same mixed state as (3),
//     except permanent: (3) closes when the descriptor arrives, whereas a
//     failed lookup never arrives and the split survived until an unrelated
//     reconnect or switch repaired it.
//
// Both doors go through the prepare/publish seam (prepareGatewayFor*, which
// dial without publishing and return the activation thunk), so these mocks
// hand back a spy thunk instead of activating on call.

// Distinct gateway identities so a listener can tell WHICH backend it was
// handed. A bare vi.fn() thunk never touches $gateway, which would let an
// out-of-order publication pass unnoticed.
const INITIAL_GATEWAY = { id: 'live-socket' }
const AGENT_GATEWAY = { id: 'agent-socket' }
const PROFILE_GATEWAY = { id: 'profile-socket' }
const activateAgent = vi.fn(() => {
  $gateway.set(AGENT_GATEWAY)

  return true
})
const activateProfile = vi.fn(() => {
  $gateway.set(PROFILE_GATEWAY)
})
const prepareGatewayForAgent = vi.fn(async (_connectionId: null | string, _profile: string) => activateAgent)
const prepareGatewayForProfile = vi.fn(async (_profile: string) => activateProfile)
const openGatewayForProfile = vi.fn(async (_profile: string) => undefined)
const $gateway = atom<unknown>(INITIAL_GATEWAY)
const resetStarmapGraph = vi.fn()

vi.mock('@/store/gateway', () => ({
  $gateway,
  openGatewayForProfile,
  prepareGatewayForAgent,
  prepareGatewayForProfile
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
  prepareGatewayForAgent.mockReset()
  prepareGatewayForAgent.mockResolvedValue(activateAgent)
  prepareGatewayForProfile.mockReset()
  prepareGatewayForProfile.mockResolvedValue(activateProfile)
  activateAgent.mockClear()
  activateProfile.mockClear()
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

    expect(prepareGatewayForAgent).toHaveBeenCalledWith('homelab', 'research')
    expect(activateAgent).toHaveBeenCalledTimes(1)
    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'homelab', profile: 'research' })
    expect($activeGatewayProfile.get()).toBe('research')
    expect($connection.get()?.mode).toBe('remote')
    expect($connection.get()?.profile).toBe('research')
  })

  it('fails the switch closed when the descriptor lookup rejects', async () => {
    // Previously this path swallowed the rejection and published anyway, which
    // left $gateway and $activeGatewayProfile on the NEW backend while
    // $connection still described the old one. Unlike the pending-descriptor
    // race below, that state did not close on its own: it survived until some
    // later reconnect or switch happened to repair it.
    getConnectionFor.mockRejectedValue(new Error('source unreachable'))

    await expect(ensureGatewayAgent('homelab', 'research')).rejects.toThrow('source unreachable')

    // Nothing published: all three still describe the previous backend, and the
    // caller can retry the switch.
    expect(activateAgent).not.toHaveBeenCalled()
    expect($activeGatewayProfile.get()).toBe('default')
    expect($connection.get()?.mode).toBe('local')
    expect($connection.get()?.profile).toBe('default')
  })

  it('does not republish a registry identity invalidated during activation', async () => {
    ensureGatewayForAgent.mockResolvedValueOnce(false)

    await ensureGatewayAgent('removed-source', 'research')

    expect($activeGatewayProfile.get()).toBe('default')
    expect($connection.get()?.mode).toBe('local')
    expect(getConnectionFor).not.toHaveBeenCalled()
  })

  it('falls through to the profile path for a null connectionId', async () => {
    getConnection.mockResolvedValue(agentConn({ mode: 'local', profile: 'research' }))

    await ensureGatewayAgent(null, 'research')

    expect(prepareGatewayForProfile).toHaveBeenCalledWith('research')
    expect(prepareGatewayForAgent).not.toHaveBeenCalled()
    expect(getConnectionFor).not.toHaveBeenCalled()
  })

  it('keeps an explicit local registry id on the registry-aware path', async () => {
    getConnectionFor.mockResolvedValue(localConn({ profile: 'research' }))

    await ensureGatewayAgent('local', 'research')

    expect(prepareGatewayForAgent).toHaveBeenCalledWith('local', 'research')
    expect(prepareGatewayForProfile).not.toHaveBeenCalled()
    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'local', profile: 'research' })
  })

  it('never publishes the agent gateway before its connection descriptor', async () => {
    // The same mixed-state window the profile path closes, through the door
    // added for the SDK's ensureAgent. A slow getConnectionFor must not leave
    // $gateway/$activeGatewayProfile on the agent's backend while $connection
    // still describes the previous one — anything requesting in that window
    // announces the WRONG mode to the new backend.
    let resolveDescriptor: (conn: HermesConnection) => void = () => undefined
    getConnectionFor.mockReturnValue(
      new Promise<HermesConnection>(resolve => {
        resolveDescriptor = resolve
      })
    )

    const switching = ensureGatewayAgent('homelab', 'research')
    // Let the socket-dial half settle; the descriptor is still pending.
    await Promise.resolve()
    await Promise.resolve()

    expect(activateAgent).not.toHaveBeenCalled()
    expect($activeGatewayProfile.get()).toBe('default')
    expect($connection.get()?.mode).toBe('local')

    resolveDescriptor(agentConn())
    await switching

    expect(activateAgent).toHaveBeenCalledTimes(1)
    expect($activeGatewayProfile.get()).toBe('research')
    expect($connection.get()?.mode).toBe('remote')
  })
})

describe('ensureGatewayAgent shares the gatewaySwitch mutex with profile switches', () => {
  it('serializes an agent activation behind an in-flight profile switch', async () => {
    const profileGate = deferred()
    const order: string[] = []

    prepareGatewayForProfile.mockImplementation(async (profile: string) => {
      order.push(`profile:${profile}`)
      await profileGate.promise

      return activateProfile
    })
    prepareGatewayForAgent.mockImplementation(async (_connectionId, profile) => {
      order.push(`agent:${profile}`)

      return activateAgent
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

    prepareGatewayForAgent.mockImplementation(async (_connectionId, profile) => {
      order.push(`agent:${profile}`)
      await agentGate.promise

      return activateAgent
    })
    prepareGatewayForProfile.mockImplementation(async (profile: string) => {
      order.push(`profile:${profile}`)

      return activateProfile
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
