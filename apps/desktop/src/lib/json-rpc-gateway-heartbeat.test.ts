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
    // connect() now resolves on gateway.ready, not on raw open — send it first.
    socket.message({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'gateway.ready', payload: { heartbeat: true } }
    })
    await connected

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
    // connect() now resolves on gateway.ready, not on raw open — send it first.
    socket.message({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'gateway.ready', payload: {} }
    })
    await connected

    await vi.advanceTimersByTimeAsync(1_000)
    expect(client.connectionState).toBe('open')
    expect(socket.readyState).toBe(FakeSocket.OPEN)
    expect(socket.sent).toEqual([])
  })
})
