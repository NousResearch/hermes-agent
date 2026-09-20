import { describe, expect, it } from 'vitest'

import { GatewayConnectError, JsonRpcGatewayClient } from './json-rpc-gateway'

/** EventTarget-based WebSocket stand-in that never opens, so each failure leg can be driven by hand. */
class StuckSocket extends EventTarget {
  readyState = 0

  send(): void {}

  close(): void {
    this.readyState = 3
  }
}

const connectErrorMessage = 'Could not connect to Hermes gateway'
const gatewayReady = JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: {} } })

const rejection = (pending: Promise<void>): Promise<Error> =>
  pending.then(
    () => new Error('connect resolved'),
    (error: Error) => error
  )

const openSocket = (socket: StuckSocket): void => {
  socket.readyState = 1
  socket.dispatchEvent(new Event('open'))
}

const dial = (connectTimeoutMs = 1000) => {
  let socket!: StuckSocket
  const sockets: StuckSocket[] = []

  const client = new JsonRpcGatewayClient({
    socketFactory: () => {
      socket = new StuckSocket()
      sockets.push(socket)

      return socket as unknown as WebSocket
    },
    heartbeatIntervalMs: 0,
    heartbeatDeadlineMs: 0,
    connectTimeoutMs,
    connectErrorMessage
  })

  const pending = client.connect('ws://gateway.test/api/ws')
  pending.catch(() => {})

  return { client, pending, socket, sockets }
}

// Regression for #41566: the overlay showed the same sentence for an auth rejection, a TLS failure and a
// silent host, so users verified the gateway with curl and still could not tell what the app hit.
describe('JsonRpcGatewayClient.connect failure classes', () => {
  it('a handshake close carries its code and reason, distinct from a timeout', async () => {
    const closed = dial()
    closed.socket.dispatchEvent(new CloseEvent('close', { code: 4403, reason: 'token_mismatch' }))
    const closeError = await rejection(closed.pending)

    const timedOut = dial(20)
    const timeoutError = await rejection(timedOut.pending)

    expect(closeError.message).toContain(connectErrorMessage)
    expect(closeError.message).toContain('4403')
    expect(closeError.message).toContain('token_mismatch')
    expect(timeoutError.message).toContain(connectErrorMessage)
    expect(timeoutError.message).toContain('20 ms')
    expect(timeoutError.message).not.toBe(closeError.message)
    expect(timedOut.client.connectionState).toBe('error')
  })

  it('every failure keeps the base message as its prefix — the overlay headline and includes() matchers bind to it', async () => {
    const { pending, socket } = dial()
    socket.dispatchEvent(new Event('error'))
    const error = await rejection(pending)

    expect(error.message.startsWith(connectErrorMessage)).toBe(true)
    expect(error.message).not.toBe(connectErrorMessage)
  })
})

describe('JsonRpcGatewayClient.connect handshake option', () => {
  it('a raw open keeps the connection pending until gateway.ready', async () => {
    const { client, pending, socket } = dial()
    let resolved = false

    void pending.then(
      () => {
        resolved = true
      },
      () => undefined
    )

    socket.dispatchEvent(new Event('open'))
    await Promise.resolve()
    await Promise.resolve()

    expect(resolved).toBe(false)
    expect(client.connectionState).toBe('connecting')

    socket.dispatchEvent(new MessageEvent('message', { data: gatewayReady }))
    await pending

    expect(resolved).toBe(true)
    expect(client.connectionState).toBe('open')
  })

  it('handshake open resolves on open and delivers a first non-ready frame to the event hub', async () => {
    let socket!: StuckSocket

    const client = new JsonRpcGatewayClient({
      socketFactory: () => (socket = new StuckSocket()) as unknown as WebSocket,
      heartbeatIntervalMs: 0,
      heartbeatDeadlineMs: 0,
      connectTimeoutMs: 1000,
      connectErrorMessage,
      handshake: 'open'
    })

    const events: string[] = []

    client.onEvent(event => events.push(event.type))

    const pending = client.connect('ws://gateway.test/api/ws')
    pending.catch(() => {})

    socket.dispatchEvent(new Event('open'))
    await pending

    expect(client.connectionState).toBe('open')

    const frame = JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'tool.start' } })
    socket.dispatchEvent(new MessageEvent('message', { data: frame }))

    expect(events).toEqual(['tool.start'])
    expect(client.connectionState).toBe('open')
  })
})

describe('JsonRpcGatewayClient.connect invalidation', () => {
  it('invalidate before raw open rejects the attempt before closed listeners redial', async () => {
    const { client, pending, socket, sockets } = dial()
    let redial: Promise<void> | undefined

    client.onState(state => {
      if (state === 'closed') {
        redial = client.connect('ws://gateway.test/api/ws')
      }
    })

    client.invalidate()

    expect(redial).toBeDefined()
    expect(redial).not.toBe(pending)
    expect(sockets).toHaveLength(2)

    const error = await rejection(pending)
    expect(error).toBeInstanceOf(GatewayConnectError)
    expect(error.message).toBe('WebSocket closed')
    expect(socket.readyState).toBe(3)
    expect(client.connectionState).toBe('connecting')

    const replacement = sockets[1]
    openSocket(replacement)
    expect(client.connectionState).toBe('connecting')
    replacement.dispatchEvent(new MessageEvent('message', { data: gatewayReady }))
    await redial

    expect(client.connectionState).toBe('open')
  })

  it('invalidate after raw open but before gateway.ready rejects the attempt and redials', async () => {
    const { client, pending, socket, sockets } = dial()
    openSocket(socket)
    expect(client.connectionState).toBe('connecting')

    client.invalidate('Connection replaced')

    const redial = client.connect('ws://gateway.test/api/ws')
    expect(redial).not.toBe(pending)
    expect(sockets).toHaveLength(2)

    const error = await rejection(pending)
    expect(error).toBeInstanceOf(GatewayConnectError)
    expect(error.message).toBe('Connection replaced')
    expect(socket.readyState).toBe(3)

    const replacement = sockets[1]
    openSocket(replacement)
    expect(client.connectionState).toBe('connecting')
    replacement.dispatchEvent(new MessageEvent('message', { data: gatewayReady }))
    await redial

    expect(client.connectionState).toBe('open')
  })
})
