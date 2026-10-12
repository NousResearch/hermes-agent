import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, describe, expect, it, vi } from 'vitest'

interface ListenerEntry {
  callback: (event: any) => void
  once: boolean
}

class FakeSocket {
  static readonly CLOSED = 3
  static readonly OPEN = 1

  readonly sent: string[] = []
  readyState = FakeSocket.OPEN
  private listeners = new Map<string, ListenerEntry[]>()

  addEventListener(type: string, callback: (event: any) => void, options?: AddEventListenerOptions): void {
    const entries = this.listeners.get(type) ?? []

    entries.push({ callback, once: Boolean(options?.once) })
    this.listeners.set(type, entries)
  }

  close(): void {
    if (this.readyState === FakeSocket.CLOSED) {
      return
    }

    this.readyState = FakeSocket.CLOSED
    this.emit('close', { code: 1000 })
  }

  emit(type: string, event: any = {}): void {
    const entries = [...(this.listeners.get(type) ?? [])]

    for (const entry of entries) {
      entry.callback(event)

      if (entry.once) {
        this.removeEventListener(type, entry.callback)
      }
    }
  }

  message(frame: unknown): void {
    this.emit('message', { data: JSON.stringify(frame) })
  }

  removeEventListener(type: string, callback: (event: any) => void): void {
    this.listeners.set(
      type,
      (this.listeners.get(type) ?? []).filter(entry => entry.callback !== callback)
    )
  }

  send(payload: string): void {
    this.sent.push(payload)
  }
}

describe('JsonRpcGatewayClient heartbeat recovery', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('invalidates a silently dead advertised socket and ignores its late frames', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', { OPEN: FakeSocket.OPEN })
    const socket = new FakeSocket()
    const states: string[] = []
    const events: string[] = []

    const client = new JsonRpcGatewayClient({
      heartbeatDeadlineMs: 45,
      heartbeatIntervalMs: 15,
      socketFactory: () => socket as unknown as WebSocket
    })

    client.onState(state => states.push(state))
    client.onAny(event => events.push(event.type))

    const connected = client.connect('ws://gateway.test/api/ws')
    socket.emit('open')
    await connected
    socket.message({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'gateway.ready', payload: { heartbeat: true } }
    })

    await vi.advanceTimersByTimeAsync(61)

    expect(client.connectionState).toBe('closed')
    expect(socket.readyState).toBe(FakeSocket.CLOSED)
    expect(states.at(-1)).toBe('closed')

    socket.message({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.complete', payload: { text: 'late duplicate' } }
    })
    expect(events).toEqual(['gateway.ready'])
  })

  it('keeps older backends open when gateway.ready does not advertise heartbeat', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', { OPEN: FakeSocket.OPEN })
    const socket = new FakeSocket()

    const client = new JsonRpcGatewayClient({
      heartbeatDeadlineMs: 45,
      heartbeatIntervalMs: 15,
      socketFactory: () => socket as unknown as WebSocket
    })

    const connected = client.connect('ws://gateway.test/api/ws')
    socket.emit('open')
    await connected
    socket.message({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'gateway.ready', payload: {} }
    })

    await vi.advanceTimersByTimeAsync(1_000)
    expect(client.connectionState).toBe('open')
    expect(socket.readyState).toBe(FakeSocket.OPEN)
    // The one frame on the wire is the client.capabilities advertisement every gateway.ready triggers.
    const methods = socket.sent.map(text => (JSON.parse(text) as { method: string }).method)

    expect(methods).toEqual(['client.capabilities'])
  })

  // A hidden window's intensive wake-up throttling fires the 15 s heartbeat
  // once a minute; the backend answers each ping at once, so the newest pong
  // is ~60 s old at every tick. Only an UNANSWERED ping is peer silence.
  it('keeps an answered socket open across throttled 60 s ticks and drops it once a ping goes unanswered', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', { OPEN: FakeSocket.OPEN })
    const socket = new FakeSocket()

    const client = new JsonRpcGatewayClient({
      heartbeatDeadlineMs: 45,
      heartbeatIntervalMs: 15,
      socketFactory: () => socket as unknown as WebSocket
    })

    const connected = client.connect('ws://gateway.test/api/ws')
    socket.emit('open')
    await connected
    socket.message({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: { heartbeat: true } } })

    const pings = () =>
      socket.sent
        .map(text => JSON.parse(text) as { id: string; method: string })
        .filter(f => f.method === 'gateway.ping')

    // One throttled tick: the clock moves 60 with only the last interval run.
    const throttledTick = async () => {
      vi.setSystemTime(Date.now() + 45)
      await vi.advanceTimersByTimeAsync(15)
    }

    for (let i = 0; i < 5; i++) {
      await throttledTick()
      socket.message({ id: pings().at(-1)!.id, jsonrpc: '2.0', result: { ok: true } })
    }

    expect(pings()).toHaveLength(5)
    expect(client.connectionState).toBe('open')

    await throttledTick()
    await throttledTick()
    expect(client.connectionState).toBe('closed')
  })
})
