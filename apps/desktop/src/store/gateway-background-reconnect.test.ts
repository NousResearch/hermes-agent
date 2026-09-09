import { type GatewayEvent } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { HermesGateway } from '@/api/client'

import {
  activeGateway,
  activeGatewayConnectionId,
  activeGatewayProfileKey,
  closeSecondaryGateways,
  configureGatewayRegistry,
  disposeSecondariesForConnection,
  liveSecondaryConnectionIds,
  openGatewayForAgent,
  pruneSecondaryGateways,
  setPrimaryGateway,
  setPrimaryGatewayConnectionId
} from './gateway'
import { requestForSessionProfile } from './session-request-router'

// Real request router, registry and JSON-RPC client; only the browser network
// boundary and Electron descriptor lookup are simulated. No backend/LLM calls.
const owner = { connectionId: 'h2-remote', mode: 'remote' as const, profile: 'research' }
const sessionId = 'h2-running-session'
const remoteUrl = 'wss://h2-remote.invalid/api/ws?profile=research'
const sockets: NetworkSocket[] = []
interface SequencedEvent extends GatewayEvent {
  seq: number
}

const history: SequencedEvent[] = []
const onEvent = vi.fn<(event: GatewayEvent) => void>()
const onActiveRouteChanged = vi.fn()
const ambient = vi.fn<() => Promise<never>>()
let primary: HermesGateway

class NetworkSocket extends EventTarget {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3
  readyState = NetworkSocket.CONNECTING
  readonly requests: Array<{ id: number; method: string; params: Record<string, unknown> }> = []

  constructor(readonly url: string) {
    super()
    sockets.push(this)
    setTimeout(() => {
      if (this.readyState !== NetworkSocket.CONNECTING) {
        return
      }

      this.readyState = NetworkSocket.OPEN
      this.dispatchEvent(new Event('open'))
    }, 1)
  }

  send(data: string): void {
    if (this.readyState !== NetworkSocket.OPEN) {
      throw new Error('send on closed mock socket')
    }

    const request = JSON.parse(data) as (typeof this.requests)[number]
    this.requests.push(request)

    const result =
      request.method === 'session.events.since'
        ? {
            events: history.filter(
              event => event.session_id === request.params.session_id && event.seq! > Number(request.params.last_seen)
            )
          }
        : { status: 'streaming' }

    // Server ACK is asynchronous, after request() installed its pending entry.
    setTimeout(() => this.receive({ jsonrpc: '2.0', id: request.id, result }), 1)
  }

  receive(frame: unknown): void {
    if (this.readyState !== NetworkSocket.OPEN) {
      return
    }

    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame) }))
  }

  close(): void {
    if (this.readyState >= NetworkSocket.CLOSING) {
      return
    }

    this.readyState = NetworkSocket.CLOSING
    setTimeout(() => this.finishClose(1000), 1)
  }

  networkDrop(withError = false): void {
    // A network-originated close is an asynchronous browser event, not a
    // synchronous call to the registry's close/dispose machinery.
    setTimeout(() => {
      if (withError) {
        this.dispatchEvent(new Event('error'))
      }

      this.finishClose(1006)
    }, 1)
  }

  private finishClose(code: number): void {
    if (this.readyState === NetworkSocket.CLOSED) {
      return
    }

    this.readyState = NetworkSocket.CLOSED
    this.dispatchEvent(new CloseEvent('close', { code, wasClean: code === 1000 }))
  }
}

function publish(type: GatewayEvent['type'], payload: Record<string, unknown>): GatewayEvent {
  const event: SequencedEvent = { type, session_id: sessionId, seq: history.length + 1, payload }
  history.push(event)

  for (const socket of sockets.filter(socket => socket.url === remoteUrl)) {
    socket.receive({ jsonrpc: '2.0', method: 'event', params: event })
  }

  return event
}

beforeEach(async () => {
  vi.useFakeTimers()
  vi.spyOn(Math, 'random').mockReturnValue(0.5)
  vi.stubGlobal('WebSocket', NetworkSocket)
  sockets.length = 0
  history.length = 0
  vi.clearAllMocks()
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    getConnectionFor: vi.fn(async () => ({
      ...owner,
      authMode: 'token',
      token: 'test-only',
      wsUrl: remoteUrl,
      baseUrl: 'https://h2-remote.invalid'
    })),
    getGatewayWsUrlFor: vi.fn(async () => remoteUrl)
  }
  configureGatewayRegistry({
    activeConnectionId: () => 'local',
    foregroundScopes: () => new Set(),
    onEvent,
    onActiveRouteChanged
  })
  primary = new HermesGateway()
  const connected = primary.connect('wss://h2-local.invalid/api/ws')
  await vi.advanceTimersByTimeAsync(1)
  await connected
  setPrimaryGateway(primary, 'default')
  setPrimaryGatewayConnectionId('local')
  onActiveRouteChanged.mockClear()
})

afterEach(() => {
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

describe('background remote turn reconnect', () => {
  it.each([
    { name: 'turn lease only, network close', retained: false, drop: true, error: false },
    { name: 'turn lease only, network error and close', retained: false, drop: true, error: true },
    { name: 'retained control, network close', retained: true, drop: true, error: false },
    { name: 'turn lease only, uninterrupted control', retained: false, drop: false, error: false }
  ])('$name: delivers completion without selecting the remote', async ({ retained, drop, error }) => {
    if (retained) {
      const opened = openGatewayForAgent(owner.connectionId, owner.profile)
      await vi.advanceTimersByTimeAsync(10)
      await opened
    }

    // No open/ensure/relay/foreground pin in the H2 case. The routed submit
    // itself acquires a whole-turn lease and releases its per-request lease
    // after ACK, leaving ONLY the whole-turn lease while the model runs.
    const submitted = requestForSessionProfile(owner, ambient, 'prompt.submit', {
      session_id: sessionId,
      text: 'Continue the already-bound background session'
    })

    await vi.advanceTimersByTimeAsync(10)
    await expect(submitted).resolves.toEqual({ status: 'streaming' })
    const remote = sockets.filter(socket => socket.url === remoteUrl)
    expect(remote).toHaveLength(1)
    expect(remote[0].requests).toEqual([
      expect.objectContaining({ method: 'prompt.submit', params: expect.objectContaining({ session_id: sessionId }) })
    ])
    expect(liveSecondaryConnectionIds()).toEqual(new Set([owner.connectionId]))
    const started = publish('message.start', {})
    expect(onEvent).toHaveBeenCalledWith(expect.objectContaining({ ...started, connectionId: owner.connectionId }))
    onEvent.mockClear()

    if (drop) {
      remote[0].networkDrop(error)
      await vi.advanceTimersByTimeAsync(1)
      expect(remote[0].readyState).toBe(NetworkSocket.CLOSED)
      // Emitted during the outage: only session.events.since can recover it.
      publish('message.delta', { text: 'during outage' })
      // First backoff is deterministic; this also crosses the backoff cap.
      // No explicit reconnect, request, selection or prewarm rescues the entry.
      await vi.advanceTimersByTimeAsync(16_000)
      await vi.dynamicImportSettled()
      await vi.advanceTimersByTimeAsync(10)
      expect
        .soft(
          sockets.filter(socket => socket.url === remoteUrl),
          'automatic redial'
        )
        .toHaveLength(2)
      expect
        .soft(liveSecondaryConnectionIds(), 'remote registry route remains live')
        .toEqual(new Set([owner.connectionId]))
      expect
        .soft(onEvent, 'outage notification is replayed')
        .toHaveBeenCalledWith(
          expect.objectContaining({ type: 'message.delta', session_id: sessionId, connectionId: owner.connectionId })
        )
    }

    const completed = publish('message.complete', { text: 'finished' })
    expect
      .soft(onEvent, 'live completion reaches the background event fan-in')
      .toHaveBeenCalledWith(
        expect.objectContaining({ ...completed, connectionId: owner.connectionId, profile: owner.profile })
      )
    expect(activeGateway()).toBe(primary)
    expect(activeGatewayProfileKey()).toBe('default')
    expect(activeGatewayConnectionId()).toBe('local')
    expect(onActiveRouteChanged).not.toHaveBeenCalled()
    expect(ambient).not.toHaveBeenCalled()

    // Normal authoritative settlement still releases an unpinned live turn.
    publish('session.info', { running: false })
    await vi.advanceTimersByTimeAsync(501)

    if (!retained) {
      expect(liveSecondaryConnectionIds()).toEqual(new Set())
      expect(
        sockets.filter(socket => socket.url === remoteUrl).every(socket => socket.readyState === NetworkSocket.CLOSED)
      ).toBe(true)
    }
  })

  it.each(['close', 'remove', 'prune', 'missing connection'] as const)(
    '%s stops reconnect and releases the old session lease before reuse',
    async cleanup => {
      const submit = () =>
        requestForSessionProfile(owner, ambient, 'prompt.submit', {
          session_id: sessionId,
          text: 'background work'
        })

      const submitted = submit()

      await vi.advanceTimersByTimeAsync(10)
      await submitted
      publish('message.start', {})
      const original = sockets.find(socket => socket.url === remoteUrl)!
      original.networkDrop()
      await vi.advanceTimersByTimeAsync(1)

      if (cleanup === 'close') {
        closeSecondaryGateways()
      } else if (cleanup === 'remove') {
        disposeSecondariesForConnection(owner.connectionId)
      } else if (cleanup === 'prune') {
        pruneSecondaryGateways(new Set())
      } else {
        vi.mocked(window.hermesDesktop!.getConnectionFor!).mockRejectedValueOnce(
          new Error(`No connection with id "${owner.connectionId}"`)
        )
      }

      await vi.advanceTimersByTimeAsync(60_000)
      expect(liveSecondaryConnectionIds()).toEqual(new Set())
      expect(sockets.filter(socket => socket.url === remoteUrl)).toEqual([original])

      // Reusing the same scope/session must acquire a new lease, not find a
      // stale key whose release closure still belongs to the disposed entry.
      const resubmitted = submit()
      await vi.advanceTimersByTimeAsync(10)
      await resubmitted
      const replacement = sockets.filter(socket => socket.url === remoteUrl)[1]
      expect(replacement.readyState).toBe(NetworkSocket.OPEN)
      publish('message.start', {})
      publish('message.complete', {})
      publish('session.info', { running: false })
      await vi.advanceTimersByTimeAsync(501)
      expect(replacement.readyState).toBe(NetworkSocket.CLOSED)
      expect(liveSecondaryConnectionIds()).toEqual(new Set())
      await vi.advanceTimersByTimeAsync(60_000)
      expect(sockets.filter(socket => socket.url === remoteUrl)).toHaveLength(2)
    }
  )
})
