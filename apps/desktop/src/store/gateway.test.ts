import type { ConnectionState, GatewayEvent } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

interface FakeGateway {
  readonly connectionState: ConnectionState
  close: ReturnType<typeof vi.fn>
  connect: ReturnType<typeof vi.fn>
  connectImplementation: (url: string) => Promise<void>
  request: ReturnType<typeof vi.fn>
  requestImplementation: (
    method: string,
    params?: Record<string, unknown>,
    timeoutMs?: number,
    signal?: AbortSignal
  ) => Promise<unknown>
  setState(state: ConnectionState): void
}

const fakes = vi.hoisted(() => ({ instances: [] as FakeGateway[] }))

vi.mock('@/hermes', () => {
  class FakeHermesGateway implements FakeGateway {
    private state: ConnectionState = 'idle'
    private readonly eventHandlers = new Set<(event: GatewayEvent) => void>()
    private readonly stateHandlers = new Set<(state: ConnectionState) => void>()

    connectImplementation: FakeGateway['connectImplementation'] = async (_url: string) => undefined
    requestImplementation: FakeGateway['requestImplementation'] = async () => undefined

    close = vi.fn(() => this.setState('closed'))
    connect = vi.fn(async (url: string) => {
      this.setState('connecting')
      await this.connectImplementation(url)
      this.setState('open')
    })
    request = vi.fn((method: string, params?: Record<string, unknown>, timeoutMs?: number, signal?: AbortSignal) =>
      this.requestImplementation(method, params, timeoutMs, signal)
    )

    constructor() {
      fakes.instances.push(this)
    }

    get connectionState(): ConnectionState {
      return this.state
    }

    onEvent(handler: (event: GatewayEvent) => void): () => void {
      this.eventHandlers.add(handler)

      return () => this.eventHandlers.delete(handler)
    }

    onState(handler: (state: ConnectionState) => void): () => void {
      this.stateHandlers.add(handler)
      handler(this.state)

      return () => this.stateHandlers.delete(handler)
    }

    setState(state: ConnectionState): void {
      this.state = state

      for (const handler of this.stateHandlers) {
        handler(state)
      }
    }
  }

  return { HermesGateway: FakeHermesGateway }
})

import { HermesGateway } from '@/hermes'

import {
  acquireGatewayRequestLease,
  closeSecondaryGateways,
  ensureGatewayForProfile,
  openGatewayForProfile,
  pruneSecondaryGateways,
  setPrimaryGateway
} from './gateway'

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

const connection = (profile: string) =>
  ({ authMode: 'token', profile, wsUrl: `ws://127.0.0.1/${profile}` }) as Awaited<
    ReturnType<NonNullable<typeof window.hermesDesktop>['getConnection']>
  >

const getConnection = vi.fn(async (profile?: string | null) => connection(profile ?? 'default'))

function latestGateway(): FakeGateway {
  const gateway = fakes.instances.at(-1)

  if (!gateway) {
    throw new Error('expected a gateway instance')
  }

  return gateway
}

async function createSecondary(profile = 'source'): Promise<FakeGateway> {
  await openGatewayForProfile(profile)

  return latestGateway()
}

beforeEach(async () => {
  closeSecondaryGateways()
  setPrimaryGateway(null)
  fakes.instances.length = 0
  getConnection.mockClear()
  vi.stubGlobal('window', {
    hermesDesktop: {
      getConnection
    }
  })

  const primary = new HermesGateway() as unknown as FakeGateway
  primary.setState('open')
  setPrimaryGateway(primary as unknown as HermesGateway, 'default')
  await ensureGatewayForProfile('default')
})

afterEach(() => {
  closeSecondaryGateways()
  setPrimaryGateway(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('profile-scoped gateway request leases', () => {
  it('recovers a disconnected primary on that exact transport before retrying', async () => {
    const primary = fakes.instances[0]
    const lease = acquireGatewayRequestLease(primary as unknown as HermesGateway, 'default')
    let attempts = 0

    primary.requestImplementation = async () => {
      attempts += 1

      if (attempts === 1) {
        primary.setState('closed')
        throw new Error('Hermes gateway connection closed')
      }

      return { transport: 'primary' }
    }

    getConnection.mockClear()
    primary.connect.mockClear()

    await expect(lease.request('session.branch', { session_id: 'primary-runtime' })).resolves.toEqual({
      transport: 'primary'
    })
    expect(getConnection).toHaveBeenCalledOnce()
    expect(getConnection).toHaveBeenCalledWith('default')
    expect(primary.connect).toHaveBeenCalledOnce()
    expect(primary.request).toHaveBeenCalledTimes(2)

    lease.release()
  })

  it('survives production pruning, reconnects once, retries the exact request, and becomes prunable on release', async () => {
    const source = await createSecondary()
    const lease = acquireGatewayRequestLease(source as unknown as HermesGateway, 'source')
    const params = { count: 2, session_id: 'runtime-source' }
    const signal = new AbortController().signal
    let attempts = 0

    source.requestImplementation = async () => {
      attempts += 1

      if (attempts === 1) {
        source.setState('closed')
        throw new Error('Hermes gateway connection closed')
      }

      return { ok: true }
    }

    await ensureGatewayForProfile('other')
    pruneSecondaryGateways(new Set())
    getConnection.mockClear()
    source.connect.mockClear()

    await expect(lease.request('session.branch', params, 1234, signal)).resolves.toEqual({ ok: true })

    expect(getConnection).toHaveBeenCalledOnce()
    expect(getConnection).toHaveBeenCalledWith('source')
    expect(source.connect).toHaveBeenCalledOnce()
    expect(source.request).toHaveBeenCalledTimes(2)
    expect(source.request.mock.calls[0]).toEqual(['session.branch', params, 1234, signal])
    expect(source.request.mock.calls[1]).toEqual(['session.branch', params, 1234, signal])

    lease.release()
    pruneSecondaryGateways(new Set())

    expect(source.close).toHaveBeenCalledOnce()
    await expect(lease.request('session.branch', params)).rejects.toThrow('Hermes source gateway unavailable')
  })

  it('shares one reconnect across concurrent leased requests instead of starting a reconnect storm', async () => {
    const source = await createSecondary()
    const firstLease = acquireGatewayRequestLease(source as unknown as HermesGateway, 'source')
    const secondLease = acquireGatewayRequestLease(source as unknown as HermesGateway, 'source')
    const reconnect = deferred<void>()
    let requestCalls = 0

    source.requestImplementation = async () => {
      requestCalls += 1

      if (requestCalls <= 2) {
        source.setState('closed')
        throw new Error('gateway not connected')
      }

      return requestCalls
    }

    source.connectImplementation = async () => reconnect.promise
    getConnection.mockClear()
    source.connect.mockClear()

    const first = firstLease.request<number>('session.branch', { session_id: 'one' })
    const second = secondLease.request<number>('session.branch', { session_id: 'two' })

    await vi.waitFor(() => {
      expect(getConnection).toHaveBeenCalledOnce()
      expect(source.connect).toHaveBeenCalledOnce()
    })

    reconnect.resolve()

    await expect(Promise.all([first, second])).resolves.toEqual([3, 4])
    expect(source.request).toHaveBeenCalledTimes(4)

    firstLease.release()
    secondLease.release()
  })

  it('does not reconnect or retry after a non-disconnect request error', async () => {
    const source = await createSecondary()
    const lease = acquireGatewayRequestLease(source as unknown as HermesGateway, 'source')
    const terminal = new Error('session branch rejected')

    source.requestImplementation = async () => {
      throw terminal
    }

    getConnection.mockClear()
    source.connect.mockClear()

    await expect(lease.request('session.branch', { session_id: 'source' })).rejects.toBe(terminal)
    expect(source.request).toHaveBeenCalledOnce()
    expect(getConnection).not.toHaveBeenCalled()
    expect(source.connect).not.toHaveBeenCalled()

    lease.release()
  })

  it('propagates the exact terminal retry error without a third attempt', async () => {
    const source = await createSecondary()
    const lease = acquireGatewayRequestLease(source as unknown as HermesGateway, 'source')
    const terminal = new Error('branch rejected after reconnect')
    let attempts = 0

    source.requestImplementation = async () => {
      attempts += 1

      if (attempts === 1) {
        source.setState('closed')
        throw new Error('gateway not connected')
      }

      throw terminal
    }

    getConnection.mockClear()
    source.connect.mockClear()

    await expect(lease.request('session.branch', { session_id: 'source' })).rejects.toBe(terminal)
    expect(source.request).toHaveBeenCalledTimes(2)
    expect(getConnection).toHaveBeenCalledOnce()
    expect(source.connect).toHaveBeenCalledOnce()

    lease.release()
  })

  it('fails closed when teardown removes the owner while reconnect metadata is pending', async () => {
    const source = await createSecondary()
    const lease = acquireGatewayRequestLease(source as unknown as HermesGateway, 'source')
    const lookup = deferred<ReturnType<typeof connection>>()

    source.requestImplementation = async () => {
      source.setState('closed')
      throw new Error('gateway not connected')
    }

    getConnection.mockImplementationOnce(async () => lookup.promise)
    getConnection.mockClear()
    source.connect.mockClear()

    const request = lease.request('session.branch', { session_id: 'source' })

    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledWith('source'))
    closeSecondaryGateways()
    lookup.resolve(connection('source'))

    await expect(request).rejects.toThrow('gateway not connected')
    expect(source.connect).not.toHaveBeenCalled()
    expect(source.request).toHaveBeenCalledOnce()

    lease.release()
  })
})
