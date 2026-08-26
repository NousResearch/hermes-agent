import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Fail-closed owner resolution for the window's ONE session-scoped RPC
// dispatcher. A request that names a session whose owner NO rung can name
// (tile route → exact hint → connection-tagged / profiled row → REST probe)
// must not ride the ambient presentation socket: "active" has no routing
// authority, and the fallback turned missing ownership metadata into a
// misleading backend "session not found". The single exception is the legacy
// single-backend Desktop (no registry source, ≤1 profile), where the ambient
// gateway IS the owner by construction.

const gatewayMocks = vi.hoisted(() => ({
  activeConnectionId: null as null | string,
  requestGatewayForAgent: vi.fn(async () => ({ routed: true })),
  requestGatewayForProfile: vi.fn(async () => ({ profiled: true }))
}))

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  activeGatewayConnectionId: () => gatewayMocks.activeConnectionId,
  requestGatewayForAgent: gatewayMocks.requestGatewayForAgent,
  requestGatewayForProfile: gatewayMocks.requestGatewayForProfile
}))

const probe = vi.hoisted(() => ({ resolveSessionOwner: vi.fn(async () => undefined as unknown) }))

vi.mock('@/app/session/hooks/use-session-actions/utils', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  resolveSessionOwner: probe.resolveSessionOwner
}))

const { createSessionRpcDispatcher } = await import('./session-rpc-dispatcher')
const { $connectionsRegistry } = await import('@/store/connection-registry-state')
const { $profiles } = await import('@/store/profile')
const { _resetSessionOwnerHintsForTests, setSessionOwnerHint, setSessions } = await import('@/store/session')
const { isSessionOwnerResolutionError } = await import('@/store/session-owner-resolution')
const { $sessionTiles } = await import('@/store/session-states')
const { makeSessionInfo } = await import('@/test/session-info')

function dispatcher(ambientRequest = vi.fn(async () => ({ ambient: true }))) {
  return {
    ambientRequest,
    request: createSessionRpcDispatcher({
      ambientRequest: ambientRequest as never,
      runtimeIdByStoredSessionIdRef: { current: new Map([['stored-omar', 'rt-omar']]) },
      selectedStoredSessionIdRef: { current: null },
      sessionStateByRuntimeIdRef: { current: new Map() }
    })
  }
}

beforeEach(() => {
  gatewayMocks.activeConnectionId = 'local'
  $connectionsRegistry.set({ connections: [{ id: 'local' }] } as never)
  $profiles.set([{ name: 'default' }, { name: 'omar' }] as never)
  probe.resolveSessionOwner.mockResolvedValue(undefined)
})

afterEach(() => {
  $connectionsRegistry.set(null)
  setSessions([])
  $sessionTiles.set([])
  $profiles.set([])
  _resetSessionOwnerHintsForTests({ storage: true })
  vi.clearAllMocks()
})

describe('createSessionRpcDispatcher: fail closed', () => {
  it('rejects with an explicit owner-resolution error instead of riding the ambient socket', async () => {
    const { ambientRequest, request } = dispatcher()

    await expect(request('prompt.submit', { session_id: 'rt-orphan', text: 'hi' })).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )
    await expect(request('prompt.submit', { session_id: 'rt-orphan', text: 'hi' })).rejects.toThrow(
      /owner could not be resolved for "rt-orphan" \(prompt.submit\)/
    )

    expect(probe.resolveSessionOwner).toHaveBeenCalledWith('rt-orphan')
    expect(ambientRequest).not.toHaveBeenCalled()
    expect(gatewayMocks.requestGatewayForAgent).not.toHaveBeenCalled()
    expect(gatewayMocks.requestGatewayForProfile).not.toHaveBeenCalled()
  })

  it('still lets a request with NO session (ambient chrome) reach the ambient socket', async () => {
    const { ambientRequest, request } = dispatcher()

    await expect(request('config.get', {})).resolves.toEqual({ ambient: true })
    expect(ambientRequest).toHaveBeenCalledWith('config.get', {})
  })

  it('keeps the legacy single-backend Desktop on the ambient socket: no registry source, one profile', async () => {
    gatewayMocks.activeConnectionId = null
    $connectionsRegistry.set(null)
    $profiles.set([{ name: 'default' }] as never)
    const { ambientRequest, request } = dispatcher()

    await expect(request('session.resume', { session_id: 'stored-legacy' })).resolves.toEqual({ ambient: true })
    expect(ambientRequest).toHaveBeenCalledWith('session.resume', { session_id: 'stored-legacy' })
  })

  it('fails closed as soon as there is somewhere to misroute to: a second profile, or a live registry source', async () => {
    gatewayMocks.activeConnectionId = null
    $connectionsRegistry.set(null)
    $profiles.set([{ name: 'default' }, { name: 'omar' }] as never)
    await expect(dispatcher().request('session.resume', { session_id: 'stored-x' })).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )

    gatewayMocks.activeConnectionId = 'local'
    $connectionsRegistry.set({ connections: [{ id: 'local' }] } as never)
    $profiles.set([{ name: 'default' }] as never)
    await expect(dispatcher().request('session.resume', { session_id: 'stored-x' })).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )
  })
})

describe('createSessionRpcDispatcher: exact owner rungs', () => {
  it('routes by the connection-tagged row when the hint is gone (runtime id translated to the stored id)', async () => {
    setSessions([makeSessionInfo({ connection_id: 'local', id: 'stored-omar', profile: 'omar' })])
    const { ambientRequest, request } = dispatcher()

    await expect(request('prompt.submit', { session_id: 'rt-omar', text: 'again' })).resolves.toEqual({ routed: true })

    expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledWith('local', 'omar', 'prompt.submit', {
      session_id: 'rt-omar',
      text: 'again'
    })
    expect(ambientRequest).not.toHaveBeenCalled()
    expect(probe.resolveSessionOwner).not.toHaveBeenCalled()
  })

  it('prefers the exact hint over an untagged row profile, and the probe result over nothing', async () => {
    setSessions([makeSessionInfo({ id: 'stored-omar', profile: 'default' })])
    setSessionOwnerHint('stored-omar', { connectionId: 'local', profile: 'omar' })

    await expect(dispatcher().request('session.interrupt', { session_id: 'rt-omar' })).resolves.toEqual({
      routed: true
    })
    expect(gatewayMocks.requestGatewayForAgent).toHaveBeenLastCalledWith('local', 'omar', 'session.interrupt', {
      session_id: 'rt-omar'
    })

    _resetSessionOwnerHintsForTests()
    setSessions([])
    probe.resolveSessionOwner.mockResolvedValue({ connectionId: 'homelab', profile: 'worker' })

    await expect(dispatcher().request('session.activate', { session_id: 'stored-hidden' })).resolves.toEqual({
      routed: true
    })
    expect(gatewayMocks.requestGatewayForAgent).toHaveBeenLastCalledWith('homelab', 'worker', 'session.activate', {
      session_id: 'stored-hidden'
    })
  })
})
