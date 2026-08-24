import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, describe, expect, it, vi } from 'vitest'

interface ListenerEntry {
  callback: (event: any) => void
  once: boolean
}

class LoopbackSocket {
  static readonly CLOSED = 3
  static readonly CONNECTING = 0
  static readonly OPEN = 1

  readyState = LoopbackSocket.CONNECTING
  private listeners = new Map<string, ListenerEntry[]>()

  constructor(
    readonly generation: number,
    private readonly backend: RecoveryBackend
  ) {}

  addEventListener(type: string, callback: (event: any) => void, options?: AddEventListenerOptions): void {
    const entries = this.listeners.get(type) ?? []

    entries.push({ callback, once: Boolean(options?.once) })
    this.listeners.set(type, entries)
  }

  close(): void {
    if (this.readyState === LoopbackSocket.CLOSED) {
      return
    }

    this.readyState = LoopbackSocket.CLOSED
    this.emit('close', { code: 1000 })
  }

  message(frame: unknown): void {
    this.emit('message', { data: JSON.stringify(frame) })
  }

  open(): void {
    this.readyState = LoopbackSocket.OPEN
    this.emit('open', {})
    this.message({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'gateway.ready', payload: { heartbeat: true } }
    })
  }

  removeEventListener(type: string, callback: (event: any) => void): void {
    this.listeners.set(
      type,
      (this.listeners.get(type) ?? []).filter(entry => entry.callback !== callback)
    )
  }

  send(payload: string): void {
    this.backend.receive(this, JSON.parse(payload) as Record<string, any>)
  }

  private emit(type: string, event: any): void {
    const entries = [...(this.listeners.get(type) ?? [])]

    for (const entry of entries) {
      entry.callback(event)

      if (entry.once) {
        this.removeEventListener(type, entry.callback)
      }
    }
  }
}

class RecoveryBackend {
  readonly sockets: LoopbackSocket[] = []
  readonly messages: Array<{ role: 'assistant' | 'user'; content: string }> = []
  promptSubmits = 0

  createSocket(): LoopbackSocket {
    const socket = new LoopbackSocket(this.sockets.length + 1, this)

    this.sockets.push(socket)
    queueMicrotask(() => socket.open())

    return socket
  }

  receive(socket: LoopbackSocket, frame: Record<string, any>): void {
    if (socket.generation === 1 && frame.method === 'prompt.submit') {
      this.promptSubmits += 1
      this.messages.push(
        { role: 'user', content: String(frame.params?.text ?? '') },
        { role: 'assistant', content: 'persisted while transport was blackholed' }
      )

      // Accepted and persisted, but every outbound packet (including the RPC
      // response, heartbeat acknowledgement, and message.complete) is dropped.
      return
    }

    if (socket.generation === 1) {
      return
    }

    if (frame.method === 'gateway.ping') {
      socket.message({ jsonrpc: '2.0', id: frame.id, result: { ok: true } })

      return
    }

    if (frame.method === 'session.activate') {
      socket.message({
        jsonrpc: '2.0',
        id: frame.id,
        result: {
          session_id: 'runtime-1',
          session_key: 'stored-1',
          messages: this.messages,
          running: false,
          status: 'idle'
        }
      })
    }
  }
}

describe('hermes-ws-recovery-v1 silent blackhole', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('recovers the persisted final on a replacement socket without duplicating the prompt', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', { OPEN: LoopbackSocket.OPEN })
    const backend = new RecoveryBackend()

    const client = new JsonRpcGatewayClient({
      heartbeatDeadlineMs: 45,
      heartbeatIntervalMs: 15,
      socketFactory: () => backend.createSocket() as unknown as WebSocket
    })

    let epoch = 0
    let intentionalShutdown = false
    let recovered: Promise<any> | null = null

    client.onState(state => {
      if (state === 'open') {
        epoch += 1

        if (epoch > 1) {
          recovered = client.request('session.activate', { session_id: 'runtime-1' })
        }
      }

      if (state === 'closed' && !intentionalShutdown) {
        void client.connect('ws://gateway.test/api/ws')
      }
    })

    await client.connect('ws://gateway.test/api/ws')
    await Promise.resolve()

    const ambiguousSubmit = client.request('prompt.submit', { session_id: 'runtime-1', text: 'one prompt' }).then(
      () => null,
      error => error as Error
    )

    await vi.advanceTimersByTimeAsync(61)
    await vi.waitFor(() => expect(backend.sockets).toHaveLength(2))
    await vi.waitFor(() => expect(recovered).not.toBeNull())

    await expect(ambiguousSubmit).resolves.toMatchObject({
      message: 'WebSocket heartbeat acknowledgement timed out'
    })
    await expect(recovered).resolves.toMatchObject({
      messages: [
        { role: 'user', content: 'one prompt' },
        { role: 'assistant', content: 'persisted while transport was blackholed' }
      ],
      running: false,
      status: 'idle'
    })
    expect(backend.promptSubmits).toBe(1)

    intentionalShutdown = true
    client.close()
    await vi.advanceTimersByTimeAsync(1_000)
    expect(backend.sockets).toHaveLength(2)
  })
})
