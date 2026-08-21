// connect() must reject before WebSocket coerces garbage into
// `ws://<origin>/[object%20Object]` (#68250 stale-emit boot loop).

import { JsonRpcGatewayClient, JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

class FakeSocket {
  static OPEN = 1
  readyState = 0
  addEventListener = vi.fn((type: string, handler: () => void) => {
    if (type === 'open') {
      setTimeout(() => {
        this.readyState = FakeSocket.OPEN
        handler()
      }, 0)
    }
  })
  removeEventListener = vi.fn()
  close = vi.fn()
  send = vi.fn()
}

describe('JsonRpcGatewayClient connect() URL guard', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', FakeSocket) // jsdom has none; class reads WebSocket.OPEN
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('rejects a non-string IPC result object', async () => {
    const client = new JsonRpcGatewayClient()
    await expect(client.connect({ ok: true, wsUrl: 'ws://127.0.0.1:1/api/ws' } as unknown as string)).rejects.toThrow(
      /requires a ws:\/\/ or wss:\/\/ URL string, got type "object"/
    )
  })

  it('rejects a non-ws URL string', async () => {
    const client = new JsonRpcGatewayClient()
    await expect(client.connect('http://127.0.0.1:1234/api/ws')).rejects.toThrow(
      /requires a ws:\/\/ or wss:\/\/ URL string/
    )
  })

  it('rejects a malformed ws URL before opening a socket', async () => {
    const client = new JsonRpcGatewayClient()
    await expect(client.connect('ws://')).rejects.toThrow(/requires a ws:\/\/ or wss:\/\/ URL string/)
    expect(client.connectionState).toBe('idle')
  })

  it('keeps connection state idle on rejection', async () => {
    const client = new JsonRpcGatewayClient()
    await client.connect(undefined as unknown as string).catch(() => undefined)
    expect(client.connectionState).toBe('idle')
  })

  it('accepts ws:// and wss://', async () => {
    for (const url of ['ws://127.0.0.1:1234/api/ws?token=t', 'wss://gw.example.com/api/ws?ticket=t']) {
      const client = new JsonRpcGatewayClient({ socketFactory: () => new FakeSocket() as unknown as WebSocket })
      await client.connect(url)
      expect(client.connectionState).toBe('open')
    }
  })

  it('settles a closed connect and cannot poison a newer socket after its old timeout', async () => {
    vi.useFakeTimers()

    class ControlledSocket {
      static OPEN = 1
      readyState = 0
      private readonly listeners = new Map<string, Set<() => void>>()

      addEventListener = vi.fn((type: string, handler: () => void) => {
        const handlers = this.listeners.get(type) ?? new Set<() => void>()
        handlers.add(handler)
        this.listeners.set(type, handlers)
      })
      removeEventListener = vi.fn((type: string, handler: () => void) => {
        this.listeners.get(type)?.delete(handler)
      })
      close = vi.fn()
      send = vi.fn()

      open(): void {
        this.readyState = ControlledSocket.OPEN

        for (const handler of this.listeners.get('open') ?? []) {
          handler()
        }
      }
    }

    const sockets: ControlledSocket[] = []
    const client = new JsonRpcGatewayClient({
      connectTimeoutMs: 50,
      socketFactory: () => {
        const socket = new ControlledSocket()
        sockets.push(socket)

        return socket as unknown as WebSocket
      }
    })

    try {
      const stale = client.connect('ws://127.0.0.1:1234/stale')
      expect(client.connectionState).toBe('connecting')
      client.close()
      await expect(stale).rejects.toThrow('WebSocket closed')

      const current = client.connect('ws://127.0.0.1:1234/current')
      sockets[1].open()
      await expect(current).resolves.toBeUndefined()
      expect(client.connectionState).toBe('open')

      await vi.advanceTimersByTimeAsync(50)
      expect(client.connectionState).toBe('open')
    } finally {
      client.close()
      vi.useRealTimers()
    }
  })
})

describe('JsonRpcGatewayClient structured errors', () => {
  it('rejects with JsonRpcGatewayError including code and data', async () => {
    class RespondingSocket {
      static OPEN = 1
      readyState = 0
      private messageHandler: ((event: { data: string }) => void) | null = null

      addEventListener = vi.fn((type: string, handler: (event?: { data: string }) => void) => {
        if (type === 'open') {
          setTimeout(() => {
            this.readyState = RespondingSocket.OPEN
            handler()
          }, 0)
        }

        if (type === 'message') {
          this.messageHandler = handler as (event: { data: string }) => void
        }
      })
      removeEventListener = vi.fn()
      close = vi.fn()
      send = vi.fn((raw: string) => {
        const req = JSON.parse(raw) as { id: string }
        queueMicrotask(() => {
          this.messageHandler?.({
            data: JSON.stringify({
              jsonrpc: '2.0',
              id: req.id,
              error: {
                code: 4018,
                message: 'target user message is no longer in session history',
                data: { user_turn_count: 2, ordinal: 5, segment_ordinal: 3 }
              }
            })
          })
        })
      })
    }

    vi.stubGlobal('WebSocket', RespondingSocket)

    try {
      const client = new JsonRpcGatewayClient({
        requestTimeoutMs: 5_000,
        socketFactory: () => new RespondingSocket() as unknown as WebSocket
      })

      await client.connect('ws://127.0.0.1:1234/api/ws?token=t')

      await expect(client.request('prompt.submit', { session_id: 's' })).rejects.toEqual(
        expect.objectContaining({
          name: 'JsonRpcGatewayError',
          code: 4018,
          message: 'target user message is no longer in session history',
          data: { user_turn_count: 2, ordinal: 5, segment_ordinal: 3 }
        })
      )
      expect(new JsonRpcGatewayError('x')).toBeInstanceOf(Error)
    } finally {
      vi.unstubAllGlobals()
    }
  })
})
