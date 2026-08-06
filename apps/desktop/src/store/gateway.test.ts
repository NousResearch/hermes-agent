import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const gatewayMock = vi.hoisted(() => ({ instances: [] as any[] }))

vi.mock('@hermes/shared', () => ({
  resolveGatewayWsUrl: vi.fn(async (_desktop, connection) => connection.wsUrl)
}))

vi.mock('@/hermes', () => ({
  HermesGateway: class MockHermesGateway {
    connectionState = 'closed'
    close = vi.fn(() => {
      this.connectionState = 'closed'
    })
    connect = vi.fn(async () => {
      this.connectionState = 'open'
    })
    onEvent = vi.fn(() => vi.fn())
    onState = vi.fn(() => vi.fn())

    constructor() {
      gatewayMock.instances.push(this)
    }
  }
}))

vi.mock('@/store/session', () => ({
  setGatewayState: vi.fn()
}))

import { closeSecondaryGateways, ensureGatewayForProfile, rehomeSecondaryGateway, setPrimaryGateway } from './gateway'

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

describe('secondary gateway re-home', () => {
  beforeEach(() => {
    closeSecondaryGateways()
    setPrimaryGateway(null, 'default')
    gatewayMock.instances.length = 0
  })

  afterEach(() => {
    closeSecondaryGateways()
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('cancels an old deferred connection lookup before the replacement socket opens', async () => {
    const oldConnection = deferred<{ wsUrl: string }>()

    const getConnection = vi
      .fn()
      .mockImplementationOnce(() => oldConnection.promise)
      .mockResolvedValueOnce({ wsUrl: 'ws://new-profile-backend' })

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      getConnection,
      touchBackend: vi.fn(async () => undefined)
    }

    const pendingOpen = ensureGatewayForProfile('writer')
    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(1))

    const oldGateway = gatewayMock.instances[0]
    await rehomeSecondaryGateway('writer')
    const replacementGateway = gatewayMock.instances[1]

    oldConnection.resolve({ wsUrl: 'ws://stale-profile-backend' })
    await pendingOpen

    expect(oldGateway.connect).not.toHaveBeenCalled()
    expect(oldGateway.close).toHaveBeenCalledOnce()
    expect(replacementGateway.connect).toHaveBeenCalledOnce()
    expect(replacementGateway.connect).toHaveBeenCalledWith('ws://new-profile-backend')
  })
})
