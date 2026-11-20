import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { HermesGateway } from '@/api/client'

import {
  closeSecondaryGateways,
  configureGatewayRegistry,
  openGatewayForAgent,
  setPrimaryGateway,
  setPrimaryGatewayConnectionId
} from './gateway'
import { parkSecondaryReconnects, secondaryReconnectsParked, unparkSecondaryReconnects } from './gateway-park'

// The registry's park gates (#C2a): while this window is hidden, NOTHING opens a
// pool socket — not the redial ladder, and not a demand-driven open (a relay tick,
// a poll). Real registry, real JSON-RPC client; only the browser network boundary
// and the Electron descriptor lookup are simulated.
const owner = { connectionId: 'park-remote', mode: 'remote' as const, profile: 'research' }
const remoteUrl = 'wss://park-remote.invalid/api/ws?profile=research'
const sockets: ParkSocket[] = []

class ParkSocket extends EventTarget {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3
  readyState = ParkSocket.CONNECTING

  constructor(readonly url: string) {
    super()
    sockets.push(this)
    setTimeout(() => {
      if (this.readyState !== ParkSocket.CONNECTING) {
        return
      }

      this.readyState = ParkSocket.OPEN
      this.dispatchEvent(new Event('open'))
      this.receive({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: {} } })
    }, 1)
  }

  send(data: string): void {
    const request = JSON.parse(data) as { id: number }

    setTimeout(() => this.receive({ jsonrpc: '2.0', id: request.id, result: { status: 'streaming' } }), 1)
  }

  receive(frame: unknown): void {
    if (this.readyState !== ParkSocket.OPEN) {
      return
    }

    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame) }))
  }

  close(): void {
    this.readyState = ParkSocket.CLOSED
    this.dispatchEvent(new CloseEvent('close', { code: 1000, wasClean: true }))
  }

  // The OS/transport losing the socket while the user is away.
  networkDrop(): void {
    this.readyState = ParkSocket.CLOSED
    this.dispatchEvent(new CloseEvent('close', { code: 1006, wasClean: false }))
  }
}

const poolSockets = () => sockets.filter(socket => socket.url === remoteUrl)

let primary: HermesGateway

beforeEach(async () => {
  vi.useFakeTimers()
  vi.spyOn(Math, 'random').mockReturnValue(0.5)
  vi.stubGlobal('WebSocket', ParkSocket)
  sockets.length = 0
  vi.clearAllMocks()
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    getConnectionFor: vi.fn(async () => ({
      ...owner,
      authMode: 'token',
      baseUrl: 'https://park-remote.invalid',
      token: 'test-only',
      wsUrl: remoteUrl
    })),
    getGatewayWsUrlFor: vi.fn(async () => remoteUrl)
  }
  configureGatewayRegistry({
    activeConnectionId: () => 'local',
    foregroundScopes: () => new Set(),
    onEvent: () => undefined,
    // Parking is host state, exactly like `foregroundScopes` above: the window
    // owns the park moment (useGatewayBoot wires this from the same latch), and
    // the registry consults it at the two points that can dial.
    routesParked: secondaryReconnectsParked
  })
  primary = new HermesGateway()
  const connected = primary.connect('wss://park-local.invalid/api/ws')
  await vi.advanceTimersByTimeAsync(1)
  await connected
  setPrimaryGateway(primary, 'default')
  setPrimaryGatewayConnectionId('local')
})

afterEach(() => {
  unparkSecondaryReconnects()
  closeSecondaryGateways()
  primary.close()
  setPrimaryGateway(null)
  setPrimaryGatewayConnectionId(null)
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('parked window (#C2a)', () => {
  it('arms no ladder: a pool socket that dies while parked stays down', async () => {
    const opened = openGatewayForAgent(owner.connectionId, owner.profile)
    await vi.advanceTimersByTimeAsync(10)
    await opened
    expect(poolSockets()).toHaveLength(1)

    parkSecondaryReconnects()
    expect(secondaryReconnectsParked()).toBe(true)

    poolSockets()[0].networkDrop()
    await vi.advanceTimersByTimeAsync(1)
    expect(poolSockets()[0].readyState).toBe(ParkSocket.CLOSED)

    // A backgrounded window pays nothing: no backoff tick re-dials the route,
    // however long the user is away. (The RETURN dials — the hook's own suite
    // asserts that, through the window that owns the wake path.)
    await vi.advanceTimersByTimeAsync(120_000)
    await vi.dynamicImportSettled()
    await vi.advanceTimersByTimeAsync(10)
    expect(poolSockets()).toHaveLength(1)
  })

  it('opens nothing on demand either: a parked window mints no ticket for a background consumer', async () => {
    parkSecondaryReconnects()

    const demand = openGatewayForAgent(owner.connectionId, owner.profile)
    await vi.advanceTimersByTimeAsync(50)
    await demand.catch(() => undefined)

    expect(poolSockets()).toHaveLength(0)

    // And it is the PARK that held it back, not a permanent failure.
    unparkSecondaryReconnects()
    const reopened = openGatewayForAgent(owner.connectionId, owner.profile)
    await vi.advanceTimersByTimeAsync(10)
    await reopened
    expect(poolSockets()).toHaveLength(1)
  })
})
