// connect() must reject before WebSocket coerces garbage into
// `ws://<origin>/[object%20Object]` (#68250 stale-emit boot loop).

import { type GatewayEvent, isGatewayReauthRequired, JsonRpcGatewayClient, JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

type FakeEvent = {
  code?: number
  data?: string
}

type FakeListener = (event: FakeEvent) => void

const gatewayReadyFrame = JSON.stringify({
  jsonrpc: '2.0',
  method: 'event',
  params: { type: 'gateway.ready' }
})

class FakeSocket {
  static OPEN = 1
  static CLOSED = 3

  readyState = 0
  private listeners = new Map<string, Set<FakeListener>>()

  addEventListener = vi.fn((type: string, handler: FakeListener) => {
    let handlers = this.listeners.get(type)

    if (!handlers) {
      handlers = new Set()
      this.listeners.set(type, handlers)
    }

    handlers.add(handler)
  })

  removeEventListener = vi.fn((type: string, handler: FakeListener) => {
    this.listeners.get(type)?.delete(handler)
  })

  close = vi.fn(() => {
    if (this.readyState === FakeSocket.CLOSED) {
      return
    }

    this.readyState = FakeSocket.CLOSED
    this.emit('close', { code: 1005 })
  })

  send = vi.fn()

  emitOpen() {
    this.readyState = FakeSocket.OPEN
    this.emit('open', {})
  }

  emitMessage(data: string) {
    this.emit('message', { data })
  }

  emitClose(code: number) {
    this.readyState = FakeSocket.CLOSED
    this.emit('close', { code })
  }

  emitError() {
    this.emit('error', {})
  }

  private emit(type: string, event: FakeEvent) {
    for (const handler of this.listeners.get(type) ?? []) {
      handler(event)
    }
  }
}

describe('JsonRpcGatewayClient connect() URL guard', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', FakeSocket) // jsdom has none; class reads WebSocket.OPEN
  })

  afterEach(() => {
    vi.useRealTimers()
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

  it('accepts ws:// and wss:// after gateway.ready', async () => {
    for (const url of ['ws://127.0.0.1:1234/api/ws?token=t', 'wss://gw.example.com/api/ws?ticket=t']) {
      const socket = new FakeSocket()

      const client = new JsonRpcGatewayClient({
        socketFactory: () => socket as unknown as WebSocket
      })

      const connectPromise = client.connect(url)

      socket.emitOpen()
      socket.emitMessage(gatewayReadyFrame)

      await connectPromise
      expect(client.connectionState).toBe('open')
      client.close()
    }
  })

  it('keeps the connection pending and connecting after raw open', async () => {
    const socket = new FakeSocket()

    const client = new JsonRpcGatewayClient({
      socketFactory: () => socket as unknown as WebSocket
    })

    const connectPromise = client.connect('ws://127.0.0.1:1234/api/ws?token=t')
    let resolved = false

    void connectPromise.then(
      () => {
        resolved = true
      },
      () => undefined
    )

    socket.emitOpen()
    await Promise.resolve()
    await Promise.resolve()

    expect(resolved).toBe(false)
    expect(client.connectionState).toBe('connecting')

    client.close()
    await expect(connectPromise).rejects.toThrow('WebSocket closed')
  })

  it('opens on gateway.ready and dispatches the readiness event before resolving', async () => {
    const socket = new FakeSocket()

    const client = new JsonRpcGatewayClient({
      socketFactory: () => socket as unknown as WebSocket
    })

    const events: GatewayEvent[] = []
    const offEvent = client.onEvent(event => events.push(event))
    const connectPromise = client.connect('ws://127.0.0.1:1234/api/ws?token=t')

    socket.emitOpen()
    socket.emitMessage(gatewayReadyFrame)
    await connectPromise

    expect(client.connectionState).toBe('open')
    expect(events).toEqual([expect.objectContaining({ type: 'gateway.ready' })])
    offEvent()
  })

  it('rejects requests before gateway.ready without sending', async () => {
    const socket = new FakeSocket()

    const client = new JsonRpcGatewayClient({
      notConnectedErrorMessage: 'Hermes gateway is not connected',
      socketFactory: () => socket as unknown as WebSocket
    })

    const connectPromise = client.connect('ws://127.0.0.1:1234/api/ws?token=t')

    socket.emitOpen()

    await expect(client.request('session.list')).rejects.toThrow('Hermes gateway is not connected')
    expect(socket.send).not.toHaveBeenCalled()

    client.close()
    await expect(connectPromise).rejects.toThrow()
  })

  it('classifies a 4401 handshake close as requiring OAuth login', async () => {
    const socket = new FakeSocket()

    const authRejectedErrorMessage =
      'Your remote gateway session has expired. Open Settings → Gateway and click "Sign in" again.'

    const client = new JsonRpcGatewayClient({
      authRejectedErrorMessage,
      connectErrorMessage: 'Could not connect to Hermes gateway',
      socketFactory: () => socket as unknown as WebSocket
    })

    const connectPromise = client.connect('wss://gw.example.com/api/ws?ticket=stale')

    socket.emitOpen()
    socket.emitClose(4401)

    const error = await connectPromise.catch(reason => reason)

    expect(error).toEqual(
      expect.objectContaining({
        message: authRejectedErrorMessage,
        needsOauthLogin: true,
        wsCloseCode: 4401
      })
    )
    expect(isGatewayReauthRequired(error)).toBe(true)
    expect(client.connectionState).toBe('closed')
  })

  it('preserves a 4403 handshake close without classifying it as reauth', async () => {
    const socket = new FakeSocket()

    const client = new JsonRpcGatewayClient({
      connectErrorMessage: 'Could not connect to Hermes gateway',
      socketFactory: () => socket as unknown as WebSocket
    })

    const connectPromise = client.connect('wss://gw.example.com/api/ws?ticket=t')

    socket.emitOpen()
    socket.emitClose(4403)

    const error = await connectPromise.catch(reason => reason)

    expect(error).toEqual(
      expect.objectContaining({
        message: 'Could not connect to Hermes gateway',
        wsCloseCode: 4403
      })
    )
    expect(isGatewayReauthRequired(error)).toBe(false)
    expect(client.connectionState).toBe('closed')
  })

  it('rejects a non-ready first frame as a protocol failure', async () => {
    const socket = new FakeSocket()

    const client = new JsonRpcGatewayClient({
      connectErrorMessage: 'Could not connect to Hermes gateway',
      socketFactory: () => socket as unknown as WebSocket
    })

    const connectPromise = client.connect('ws://127.0.0.1:1234/api/ws?token=t')

    socket.emitOpen()
    socket.emitMessage(
      JSON.stringify({
        jsonrpc: '2.0',
        method: 'event',
        params: { type: 'session.event' }
      })
    )

    await expect(connectPromise).rejects.toThrow('Could not connect to Hermes gateway')
    expect(client.connectionState).toBe('error')
    expect(socket.close).toHaveBeenCalledOnce()
  })

  it('close() rejects an in-flight handshake and its timeout cannot poison the next connection', async () => {
    vi.useFakeTimers()

    const sockets: FakeSocket[] = []

    const client = new JsonRpcGatewayClient({
      closedErrorMessage: 'Hermes gateway connection closed',
      socketFactory: () => {
        const socket = new FakeSocket()
        sockets.push(socket)

        return socket as unknown as WebSocket
      }
    })

    const firstConnect = client.connect('ws://127.0.0.1:1234/api/ws?token=first')
    sockets[0].emitOpen()

    client.close()

    await expect(firstConnect).rejects.toThrow('Hermes gateway connection closed')
    expect(client.connectionState).toBe('closed')

    const secondConnect = client.connect('ws://127.0.0.1:1234/api/ws?token=second')
    sockets[1].emitOpen()
    sockets[1].emitMessage(gatewayReadyFrame)
    await secondConnect

    expect(client.connectionState).toBe('open')

    await vi.advanceTimersByTimeAsync(15_000)

    expect(client.connectionState).toBe('open')
  })

  it('shares a same-URL attempt and rejects a different URL while connecting', async () => {
    const socket = new FakeSocket()

    const client = new JsonRpcGatewayClient({
      socketFactory: () => socket as unknown as WebSocket
    })

    const firstConnect = client.connect('ws://127.0.0.1:1234/api/ws?token=t')
    const sameUrlConnect = client.connect('ws://127.0.0.1:1234/api/ws?token=t')

    expect(sameUrlConnect).toBe(firstConnect)

    await expect(client.connect('ws://127.0.0.1:4321/api/ws?token=t')).rejects.toThrow(
      'gateway connect() already in progress'
    )

    socket.emitOpen()
    socket.emitMessage(gatewayReadyFrame)

    await expect(Promise.all([firstConnect, sameUrlConnect])).resolves.toEqual([undefined, undefined])
    expect(client.connectionState).toBe('open')
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
            this.messageHandler?.({ data: gatewayReadyFrame })
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
