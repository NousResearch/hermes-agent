import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Connection lifecycle for registry-scoped secondary gateways:
//
//  1. Removing a connection must dispose its secondaries — remote/cloud
//     sources have no local process whose death would drop the socket, so
//     without an explicit dispose the WebSocket stays open and streams ghost
//     events until page reload.
//  2. A materially edited connection re-dials so fresh sockets target the
//     NEW endpoint.
//  3. When the Electron main reports the connection no longer exists
//     (`No connection with id`), the reconnect loop fail-stops and evicts
//     the entry instead of retrying forever.

const gatewayMocks = vi.hoisted(() => {
  const instances: { close: ReturnType<typeof vi.fn>; connectionState: string }[] = []

  return {
    connect: vi.fn(async (_wsUrl: string): Promise<void> => undefined),
    instances
  }
})

vi.mock('@/hermes', () => ({
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    close = vi.fn(() => {
      this.connectionState = 'closed'
    })
    connect = async (wsUrl: string): Promise<void> => {
      await gatewayMocks.connect(wsUrl)
      this.connectionState = 'open'
    }
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
    constructor() {
      gatewayMocks.instances.push(this as never)
    }
  }
}))
vi.mock('@/store/session', () => ({
  setConnection: vi.fn(),
  setGatewayState: vi.fn()
}))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const {
  closeSecondaryGateways,
  configureGatewayRegistry,
  disposeSecondariesForConnection,
  ensureActiveGatewayOpen,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  openGatewayForProfile,
  reconnectSecondaryGateways,
  retireLocalProfileGateways,
  setPrimaryGateway
} = await import('./gateway')

function installDesktop(stub: Record<string, unknown>): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = stub
}

function descriptorFor(connectionId: string, profile: string) {
  return {
    authMode: 'token',
    baseUrl: `https://${connectionId}.invalid`,
    mode: 'remote',
    profile,
    token: 'fake-test-token',
    wsUrl: `wss://${connectionId}.invalid/api/ws?profile=${profile}`
  }
}

beforeEach(() => {
  configureGatewayRegistry({ onEvent: vi.fn() } as never)
  setPrimaryGateway({ connectionState: 'open' } as never, 'default')
})

afterEach(() => {
  closeSecondaryGateways()
  gatewayMocks.instances.length = 0
  vi.clearAllMocks()
  vi.useRealTimers()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('disposeSecondariesForConnection', () => {
  it('closes and evicts every secondary scoped to the removed connection', async () => {
    const getConnectionFor = vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
      descriptorFor(connectionId, profile)
    )

    installDesktop({ getConnectionFor })

    await ensureGatewayForAgent('homelab', 'default')
    await ensureGatewayForAgent('homelab', 'work')
    await ensureGatewayForAgent('office', 'default')

    expect(gatewayMocks.instances).toHaveLength(3)

    disposeSecondariesForConnection('homelab')

    // Both homelab sockets closed; the office socket untouched.
    expect(gatewayMocks.instances[0].close).toHaveBeenCalledOnce()
    expect(gatewayMocks.instances[1].close).toHaveBeenCalledOnce()
    expect(gatewayMocks.instances[2].close).not.toHaveBeenCalled()

    // No redial for a removal.
    expect(getConnectionFor).toHaveBeenCalledTimes(3)
  })

  it('re-dials disposed secondaries when redial is requested (material edit)', async () => {
    const getConnectionFor = vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
      descriptorFor(connectionId, profile)
    )

    installDesktop({ getConnectionFor })

    await ensureGatewayForAgent('homelab', 'default')
    expect(gatewayMocks.connect).toHaveBeenCalledTimes(1)

    disposeSecondariesForConnection('homelab', { redial: true })

    // The redial runs async through the normal open path — flush it.
    await vi.waitFor(() => {
      expect(gatewayMocks.connect).toHaveBeenCalledTimes(2)
    })

    // Old socket closed, fresh descriptor fetched (would carry the new URL).
    expect(gatewayMocks.instances[0].close).toHaveBeenCalledOnce()
    expect(getConnectionFor).toHaveBeenCalledTimes(2)
  })

  it('is a no-op for blank or unknown connection ids', async () => {
    installDesktop({
      getConnectionFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
        descriptorFor(connectionId, profile)
      )
    })

    await ensureGatewayForAgent('homelab', 'default')

    disposeSecondariesForConnection('')
    disposeSecondariesForConnection('ghost')

    expect(gatewayMocks.instances[0].close).not.toHaveBeenCalled()
  })
})

describe('retireLocalProfileGateways', () => {
  it('retires both local profile scopes without touching the same-named remote agent', async () => {
    const getConnection = vi.fn(async (profile: string) => descriptorFor('legacy-local', profile))

    const getConnectionFor = vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
      descriptorFor(connectionId, profile)
    )

    installDesktop({ getConnection, getConnectionFor })

    await openGatewayForProfile('selena')
    await ensureGatewayForAgent('local', 'selena')
    await ensureGatewayForAgent('homelab', 'selena')

    expect(gatewayMocks.instances).toHaveLength(3)
    const connectionCallsBeforeRetire = getConnection.mock.calls.length + getConnectionFor.mock.calls.length

    retireLocalProfileGateways('selena')

    expect(gatewayMocks.instances[0].close).toHaveBeenCalledOnce()
    expect(gatewayMocks.instances[1].close).toHaveBeenCalledOnce()
    expect(gatewayMocks.instances[2].close).not.toHaveBeenCalled()

    // A wake/reconnect sweep cannot redial either retired local scope. The
    // homelab entry remains open and therefore also needs no extra dial.
    reconnectSecondaryGateways()
    await Promise.resolve()
    expect(getConnection.mock.calls.length + getConnectionFor.mock.calls.length).toBe(connectionCallsBeforeRetire)
  })

  it('allows an explicit later access to create a fresh profile secondary', async () => {
    const getConnection = vi.fn(async (profile: string) => descriptorFor('legacy-local', profile))

    installDesktop({ getConnection })

    await openGatewayForProfile('selena')
    retireLocalProfileGateways('selena')
    await openGatewayForProfile('selena')

    expect(gatewayMocks.instances).toHaveLength(2)
    expect(gatewayMocks.instances[0].close).toHaveBeenCalledOnce()
    expect(gatewayMocks.instances[1].close).not.toHaveBeenCalled()
  })
})

describe('reconnect fail-stop on a removed connection', () => {
  it('evicts the entry instead of retrying when the registry no longer knows the id', async () => {
    const getConnectionFor = vi
      .fn()
      .mockResolvedValueOnce(descriptorFor('homelab', 'default'))
      .mockRejectedValue(new Error('No connection with id "homelab".'))

    installDesktop({ getConnectionFor })

    await ensureGatewayForAgent('homelab', 'default')
    expect(gatewayMocks.instances).toHaveLength(1)

    // Simulate the socket dropping after the connection was removed.
    const socket = gatewayMocks.instances[0] as unknown as { connectionState: string }
    socket.connectionState = 'closed'

    // ensureActiveGatewayOpen drives reconnectSecondary for the active scope.
    const result = await ensureActiveGatewayOpen()

    expect(result).toBeNull()
    // Fail-stop: the entry was disposed + evicted, so a second drive finds
    // nothing to retry (no further getConnectionFor calls).
    const callsAfterFailStop = getConnectionFor.mock.calls.length
    await ensureActiveGatewayOpen()
    expect(getConnectionFor.mock.calls.length).toBe(callsAfterFailStop)
  })

  it('keeps retrying on ordinary transport failures', async () => {
    const getConnectionFor = vi
      .fn()
      .mockResolvedValueOnce(descriptorFor('homelab', 'default'))
      .mockRejectedValueOnce(new Error('ECONNREFUSED'))
      .mockResolvedValue(descriptorFor('homelab', 'default'))

    installDesktop({ getConnectionFor })

    await ensureGatewayForAgent('homelab', 'default')

    const socket = gatewayMocks.instances[0] as unknown as { connectionState: string }
    socket.connectionState = 'closed'

    // First drive fails with a transport error → entry survives.
    await ensureActiveGatewayOpen()
    // Second drive succeeds against the surviving entry.
    const reopened = await ensureActiveGatewayOpen()

    expect(reopened).not.toBeNull()
  })

  it('evicts a LOCAL profile entry when the deletion guard reports the profile gone (#88769)', async () => {
    // A stale rail badge clicked after deletion drives reconnects against
    // Electron's spawn guard, which rejects every attempt. That rejection is
    // permanent — the loop must fail-stop, not hammer the guard on backoff.
    // sharedPrimaryRoute probes getConnection too, so resolve enough calls to
    // get the socket open before the guard starts rejecting.
    let connectionCalls = 0

    const getConnection = vi.fn(async () => {
      connectionCalls += 1

      if (connectionCalls <= 3) {
        return descriptorFor('legacy-local', 'selena')
      }

      throw new Error('Profile "selena" no longer exists.')
    })

    installDesktop({ getConnection })

    await openGatewayForProfile('selena')
    await ensureGatewayForProfile('selena')
    expect(gatewayMocks.instances).toHaveLength(1)
    connectionCalls = 99

    const socket = gatewayMocks.instances[0] as unknown as { connectionState: string }
    socket.connectionState = 'closed'

    // Drive the reconnect: the guard rejection must dispose + evict.
    const result = await ensureActiveGatewayOpen()

    expect(result).toBeNull()
    const callsAfterFailStop = getConnection.mock.calls.length
    await ensureActiveGatewayOpen()
    expect(getConnection.mock.calls.length).toBe(callsAfterFailStop)
  })

  it('fail-stops on the mid-delete guard rejection too', async () => {
    let connectionCalls = 0

    const getConnection = vi.fn(async () => {
      connectionCalls += 1

      if (connectionCalls <= 3) {
        return descriptorFor('legacy-local', 'selena')
      }

      throw new Error('Profile "selena" is being deleted.')
    })

    installDesktop({ getConnection })

    await openGatewayForProfile('selena')
    await ensureGatewayForProfile('selena')
    connectionCalls = 99

    const socket = gatewayMocks.instances[0] as unknown as { connectionState: string }
    socket.connectionState = 'closed'

    await ensureActiveGatewayOpen()

    const callsAfterFailStop = getConnection.mock.calls.length
    await ensureActiveGatewayOpen()
    expect(getConnection.mock.calls.length).toBe(callsAfterFailStop)
  })

  it('waits out an in-flight secondary activation instead of failing instantly (#88880)', async () => {
    // A remote secondary whose activation is ALREADY in flight from another
    // path (wake sweep, agent activation) used to make ensureActiveGatewayOpen
    // return null immediately: reconnectSecondary early-returns on
    // `reconnecting`, the socket is still closed, and the caller surfaced
    // "Hermes gateway is not connected" on the Sessions + action. The drive
    // must ride out the in-flight activation and hand back the opened socket.
    let releaseDial: (() => void) | undefined
    const dialGate = new Promise<void>(resolve => {
      releaseDial = resolve
    })

    const getConnectionFor = vi.fn(async () => descriptorFor('homelab', 'default'))

    installDesktop({ getConnectionFor })

    gatewayMocks.connect
      .mockImplementationOnce(async () => undefined) // initial open
      .mockImplementationOnce(async () => {
        await dialGate // the sweep-driven reconnect dial hangs until released
      })

    await ensureGatewayForAgent('homelab', 'default')

    const socket = gatewayMocks.instances[0] as unknown as { connectionState: string }
    socket.connectionState = 'closed'

    // Another path (the wake sweep) starts the reconnect first — the drive
    // below meets an entry that is already `reconnecting`.
    reconnectSecondaryGateways()
    await Promise.resolve()

    const driving = ensureActiveGatewayOpen()

    await new Promise(resolve => setTimeout(resolve, 300))
    releaseDial?.()

    const result = await driving

    expect(result).not.toBeNull()
    expect((result as unknown as { connectionState: string }).connectionState).toBe('open')
  })
})
