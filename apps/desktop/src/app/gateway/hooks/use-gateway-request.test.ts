import type { GatewayWsUrlResult } from '@hermes/shared'
import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const gatewayMocks = vi.hoisted(() => ({
  instances: [] as Array<{
    connect: ReturnType<typeof vi.fn>
    connectionState: string
    request: ReturnType<typeof vi.fn>
    wsUrl: string
  }>
}))

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<typeof HermesModule>()

  class FakeHermesGateway {
    connectionState = 'closed'
    wsUrl = ''
    request = vi.fn()
    connect = vi.fn(async (wsUrl: string) => {
      this.wsUrl = wsUrl
      this.connectionState = 'open'

      for (const handler of this.stateHandlers) {
        handler('open')
      }
    })
    close = vi.fn(() => {
      this.connectionState = 'closed'

      for (const handler of this.stateHandlers) {
        handler('closed')
      }
    })
    onEvent = vi.fn(() => () => undefined)
    onState = vi.fn((handler: (state: string) => void) => {
      this.stateHandlers.add(handler)
      handler(this.connectionState)

      return () => this.stateHandlers.delete(handler)
    })
    private stateHandlers = new Set<(state: string) => void>()

    constructor() {
      gatewayMocks.instances.push(this)
    }
  }

  return { ...actual, HermesGateway: FakeHermesGateway }
})

import type * as HermesModule from '@/hermes'
import type { HermesGateway } from '@/hermes'
import {
  $gateway,
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForAgent,
  setPrimaryGateway
} from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { useGatewayRequest } from './use-gateway-request'

interface TestGateway {
  connect: ReturnType<typeof vi.fn>
  connectionState: string
  request: ReturnType<typeof vi.fn>
  wsUrl?: string
}

const fakeGateway = { connectionState: 'open' } as unknown as HermesGateway

const remoteConnection = {
  authMode: 'oauth' as const,
  baseUrl: 'https://ssh.example.test',
  connectionId: 'ssh-source',
  mode: 'remote' as const,
  profile: 'research',
  remoteIdentity: 'ssh.example.test',
  remoteKind: 'ssh' as const,
  token: 'remote-token',
  wsUrl: 'wss://ssh.example.test/api/ws?ticket=stale'
}

function installRemoteDesktop() {
  let mintCount = 0

  const getConnection = vi.fn(async (profile?: null | string) => ({
    authMode: 'token' as const,
    baseUrl: 'http://127.0.0.1:5151',
    mode: 'local' as const,
    profile: profile ?? 'default',
    token: 'local-token',
    wsUrl: 'ws://127.0.0.1:5151/api/ws?token=local'
  }))

  const getConnectionFor = vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
    ...remoteConnection,
    connectionId,
    profile
  }))

  const getGatewayWsUrl = vi.fn(async () => ({
    ok: true as const,
    wsUrl: 'ws://127.0.0.1:5151/api/ws?token=fresh-local'
  }))

  const getGatewayWsUrlFor = vi.fn(
    async ({ connectionId, profile }: { connectionId: string; profile: string }): Promise<GatewayWsUrlResult> => {
      mintCount += 1

      return {
        ok: true as const,
        wsUrl: `wss://${connectionId}.example.test/api/ws?profile=${profile}&ticket=fresh-${mintCount}`
      }
    }
  )

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { getConnection, getConnectionFor, getGatewayWsUrl, getGatewayWsUrlFor }
  })

  return { getConnection, getConnectionFor, getGatewayWsUrl, getGatewayWsUrlFor }
}

function installPrimaryDesktop(authMode: 'oauth' | 'token') {
  const getConnection = vi.fn(async (profile?: null | string) => ({
    authMode,
    baseUrl: authMode === 'oauth' ? 'https://gateway.example.test' : 'http://127.0.0.1:5151',
    mode: authMode === 'oauth' ? ('remote' as const) : ('local' as const),
    profile: profile ?? 'default',
    token: 'primary-token',
    wsUrl: authMode === 'oauth' ? 'wss://gateway.example.test/api/ws?ticket=stale' : 'ws://127.0.0.1:5151/api/ws'
  }))

  const getGatewayWsUrl = vi.fn(async (profile?: null | string) => ({
    ok: true as const,
    wsUrl:
      authMode === 'oauth'
        ? `wss://gateway.example.test/api/ws?profile=${profile ?? 'default'}&ticket=fresh`
        : 'ws://127.0.0.1:5151/api/ws?token=fresh'
  }))

  const getConnectionFor = vi.fn()
  const getGatewayWsUrlFor = vi.fn()

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { getConnection, getConnectionFor, getGatewayWsUrl, getGatewayWsUrlFor }
  })

  return { getConnection, getConnectionFor, getGatewayWsUrl, getGatewayWsUrlFor }
}

function makePrimaryGateway(): TestGateway {
  return {
    connect: vi.fn(async () => undefined),
    connectionState: 'open',
    request: vi.fn()
  }
}

async function activateRemoteGateway() {
  const desktop = installRemoteDesktop()
  const primary = makePrimaryGateway()

  setPrimaryGateway(primary as unknown as HermesGateway, 'default')
  $gateway.set(primary as unknown as HermesGateway)
  await ensureGatewayForAgent('ssh-source', 'research')

  const gateway = $gateway.get() as unknown as TestGateway

  expect(gateway).not.toBe(primary)
  expect($activeGatewayProfile.get()).toBe('research')

  return { desktop, gateway }
}

async function expectSecondaryRecoveryFailure(
  gateway: TestGateway,
  request: ReturnType<typeof useGatewayRequest>['requestGateway']
) {
  const transportError = new Error('connection closed')
  gateway.request.mockRejectedValueOnce(transportError)
  gateway.connectionState = 'closed'

  vi.useFakeTimers()

  const retry = request('session.resume').then(
    () => undefined,
    error => error
  )

  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
    await vi.advanceTimersByTimeAsync(8_000)
  })

  await expect(retry).resolves.toBe(transportError)
  expect(gateway.request).toHaveBeenCalledTimes(1)
  expect(gateway.connect).toHaveBeenCalledTimes(1)
}

beforeEach(() => {
  gatewayMocks.instances.length = 0
  closeSecondaryGateways()
  setPrimaryGateway(null)
  $gateway.set(null)
  $connection.set(null)
  $gatewayState.set('idle')
  $activeGatewayProfile.set('default')
  configureGatewayRegistry({
    onActiveRouteChanged: profile => $activeGatewayProfile.set(profile),
    onEvent: vi.fn()
  })
})

afterEach(() => {
  vi.useRealTimers()
  closeSecondaryGateways()
  setPrimaryGateway(null)
  $gateway.set(null)
  $connection.set(null)
  $gatewayState.set('idle')
  $activeGatewayProfile.set('default')
  Reflect.deleteProperty(window, 'hermesDesktop')
})

describe('useGatewayRequest', () => {
  it('exposes the live gateway on the first render, before effects run', () => {
    $gateway.set(fakeGateway)

    const { result } = renderHook(() => useGatewayRequest())

    expect(result.current.gateway).toBe(fakeGateway)
  })

  it('tracks the gateway when the active socket changes', () => {
    const { result } = renderHook(() => useGatewayRequest())

    expect(result.current.gateway).toBeNull()

    act(() => $gateway.set(fakeGateway))

    expect(result.current.gateway).toBe(fakeGateway)
  })

  it.each([
    { error: new Error('connection closed'), label: 'closed message' },
    { error: new Error('ECONNRESET'), label: 'reset message' },
    { error: Object.assign(new Error('socket failed'), { code: 'ECONNRESET' }), label: 'error code' },
    { error: Object.assign(new Error('socket failed'), { cause: { code: 'ECONNRESET' } }), label: 'cause code' }
  ])('recovers the registered remote source after a $label failure', async ({ error }) => {
    const { desktop, gateway } = await activateRemoteGateway()
    gateway.request.mockResolvedValueOnce({ turn: 1 }).mockRejectedValueOnce(error).mockResolvedValueOnce({ turn: 2 })

    const { result } = renderHook(() => useGatewayRequest())

    await act(async () => {
      await expect(result.current.requestGateway('prompt.submit', { text: 'first' })).resolves.toEqual({ turn: 1 })
    })
    gateway.connectionState = 'closed'
    await act(async () => {
      await expect(result.current.requestGateway('prompt.submit', { text: 'second' })).resolves.toEqual({ turn: 2 })
    })

    expect(desktop.getConnectionFor).toHaveBeenCalledTimes(2)
    expect(desktop.getConnectionFor).toHaveBeenCalledWith({ connectionId: 'ssh-source', profile: 'research' })
    expect(desktop.getGatewayWsUrlFor).toHaveBeenCalledTimes(2)
    expect(desktop.getGatewayWsUrlFor).toHaveBeenCalledWith({ connectionId: 'ssh-source', profile: 'research' })
    expect(desktop.getConnection).not.toHaveBeenCalled()
    expect(desktop.getGatewayWsUrl).not.toHaveBeenCalled()
    expect(gateway.connect).toHaveBeenLastCalledWith(expect.stringContaining('ticket=fresh-2'))
  })

  it('does not reconnect for a non-transport request failure', async () => {
    const { desktop, gateway } = await activateRemoteGateway()
    const failure = Object.assign(new Error('request rejected'), { code: 'EVALIDATION' })
    gateway.request.mockRejectedValueOnce(failure)

    const { result } = renderHook(() => useGatewayRequest())

    await expect(result.current.requestGateway('session.resume')).rejects.toBe(failure)
    expect(desktop.getConnectionFor).toHaveBeenCalledTimes(1)
    expect(desktop.getGatewayWsUrlFor).toHaveBeenCalledTimes(1)
    expect(gateway.connect).toHaveBeenCalledTimes(1)
  })

  it('surfaces a real secondary OAuth reauth rejection as the original transport failure', async () => {
    const { desktop, gateway } = await activateRemoteGateway()
    desktop.getGatewayWsUrlFor.mockImplementation(async () => ({
      error: '401 cookie expired',
      needsOauthLogin: true,
      ok: false as const
    }))

    const { result } = renderHook(() => useGatewayRequest())

    await expectSecondaryRecoveryFailure(gateway, result.current.requestGateway)

    expect(desktop.getConnectionFor).toHaveBeenCalledWith({ connectionId: 'ssh-source', profile: 'research' })
    expect(desktop.getGatewayWsUrlFor).toHaveBeenCalled()
    expect(desktop.getConnection).not.toHaveBeenCalled()
    expect(desktop.getGatewayWsUrl).not.toHaveBeenCalled()
  })

  it('surfaces a failed secondary OAuth ticket mint without using the stale ticket or local bridges', async () => {
    const { desktop, gateway } = await activateRemoteGateway()
    desktop.getGatewayWsUrlFor.mockRejectedValue(new Error('ticket mint failed'))

    const { result } = renderHook(() => useGatewayRequest())

    await expectSecondaryRecoveryFailure(gateway, result.current.requestGateway)

    expect(desktop.getConnectionFor).toHaveBeenCalledWith({ connectionId: 'ssh-source', profile: 'research' })
    expect(desktop.getConnection).not.toHaveBeenCalled()
    expect(desktop.getGatewayWsUrl).not.toHaveBeenCalled()
  })

  it('surfaces a missing optional scoped mint bridge without falling back to a stale ticket or local lookup', async () => {
    const { desktop, gateway } = await activateRemoteGateway()
    Reflect.deleteProperty(window.hermesDesktop, 'getGatewayWsUrlFor')

    const { result } = renderHook(() => useGatewayRequest())

    await expectSecondaryRecoveryFailure(gateway, result.current.requestGateway)

    expect(desktop.getConnectionFor).toHaveBeenCalledWith({ connectionId: 'ssh-source', profile: 'research' })
    expect(desktop.getConnection).not.toHaveBeenCalled()
    expect(desktop.getGatewayWsUrl).not.toHaveBeenCalled()
  })

  it.each([
    { authMode: 'oauth' as const, label: 'primary OAuth' },
    { authMode: 'token' as const, label: 'local primary' }
  ])('preserves $label recovery', async ({ authMode }) => {
    const desktop = installPrimaryDesktop(authMode)
    const primary = makePrimaryGateway()
    primary.request.mockRejectedValueOnce(new Error('connection closed')).mockResolvedValueOnce({ recovered: true })

    setPrimaryGateway(primary as unknown as HermesGateway, 'default')
    $gateway.set(primary as unknown as HermesGateway)
    $gatewayState.set('closed')

    const { result } = renderHook(() => useGatewayRequest())

    await act(async () => {
      await expect(result.current.requestGateway('session.resume')).resolves.toEqual({ recovered: true })
    })

    expect(desktop.getConnection).toHaveBeenCalledWith('default')
    expect(desktop.getGatewayWsUrl).toHaveBeenCalledWith('default')
    expect(desktop.getConnectionFor).not.toHaveBeenCalled()
    expect(desktop.getGatewayWsUrlFor).not.toHaveBeenCalled()
  })

  it('rejects instead of hanging forever when the reconnect getConnection() wedges (#93454)', async () => {
    // Repro: a request lands on a dropped socket, the "not connected" catch
    // kicks off a reconnect, and the IPC round-trip into main
    // (desktop.getConnection) never settles — e.g. a wedged revalidation after
    // a liveness-probe trip. Without an internal timeout on that await,
    // reconnectingRef never clears and requestGateway hangs forever instead of
    // surfacing the original transport error.
    vi.useFakeTimers()

    const dropped = {
      connectionState: 'closed',
      request: vi.fn().mockRejectedValue(new Error('connection closed'))
    } as unknown as HermesGateway

    const getConnection = vi.fn(() => new Promise(() => undefined))

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { getConnection }
    $gateway.set(dropped)

    const { result } = renderHook(() => useGatewayRequest())

    const pending = expect(result.current.requestGateway('some.method')).rejects.toThrow('connection closed')

    // Advance past the internal reconnect-attempt timeout (20s) — the stalled
    // getConnection() await must reject so the reconnect gives up and the
    // original transport error surfaces, instead of requestGateway() never
    // settling.
    await vi.advanceTimersByTimeAsync(20_000)
    await pending
  })
})
