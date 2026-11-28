import { describe, expect, it } from 'vitest'

import { JsonRpcGatewayClient } from './json-rpc-gateway'

/** EventTarget-based WebSocket stand-in that never opens, so each failure leg can be driven by hand. */
class StuckSocket extends EventTarget {
  readyState = 0

  send(): void {}

  close(): void {
    this.readyState = 3
  }
}

const connectErrorMessage = 'Could not connect to Hermes gateway'

const rejection = (pending: Promise<void>): Promise<Error> =>
  pending.then(
    () => new Error('connect resolved'),
    (error: Error) => error
  )

const dial = (connectTimeoutMs = 1000) => {
  let socket!: StuckSocket

  const client = new JsonRpcGatewayClient({
    socketFactory: () => (socket = new StuckSocket()) as unknown as WebSocket,
    heartbeatIntervalMs: 0,
    heartbeatDeadlineMs: 0,
    connectTimeoutMs,
    connectErrorMessage
  })

  const pending = client.connect('ws://gateway.test/api/ws')
  pending.catch(() => {})

  return { client, pending, socket }
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

// A local Electron dial resolves its ticket through main and deliberately falls back to the bare
// `?native_dial=unminted` descriptor when that mint fails, so the reconnect loop owns the retry
// (apps/desktop/src/lib/gateway-ws-url.test.ts, "local Electron native dial"). `openGatewaySocket`
// refuses exactly that URL synchronously. When the throw escaped connect(), the client was left in
// 'connecting' with no socket and no timer, and the `state === 'connecting'` guard turned every
// later connect() into a silent no-op — the Desktop then sat on "Backend connecting" forever,
// burning a core, opening no TCP. A factory throw is a failed dial, not a wedged client.
describe('JsonRpcGatewayClient.connect factory throw', () => {
  const ticketError = () => new Error('Native gateway requires a fresh private ticket')

  const wedged = (connectTimeoutMs = 1000) => {
    const calls: string[] = []

    const client = new JsonRpcGatewayClient({
      socketFactory: () => {
        calls.push('dial')
        throw ticketError()
      },
      heartbeatIntervalMs: 0,
      heartbeatDeadlineMs: 0,
      connectTimeoutMs,
      connectErrorMessage
    })

    return { calls, client }
  }

  it('rejects the connect instead of leaving the client wedged in connecting', async () => {
    const { client } = wedged()

    const pending = client.connect('ws://gateway.test/api/ws?native_dial=unminted')

    await expect(pending).rejects.toThrow(ticketError().message)
    expect(client.connectionState).toBe('error')
  })

  it('lets the next connect() dial again — the retry the fallback contract promises', async () => {
    const { calls, client } = wedged()

    await expect(client.connect('ws://gateway.test/api/ws?native_dial=unminted')).rejects.toThrow()

    // Still throwing is fine; being refused before the factory ran is the wedge.
    await expect(client.connect('ws://gateway.test/api/ws?native_dial=unminted')).rejects.toThrow()
    expect(calls).toEqual(['dial', 'dial'])
  })

  it('names the factory failure in the overlay headline the same way every other class does', async () => {
    const { client } = wedged()

    await expect(client.connect('ws://gateway.test/api/ws?native_dial=unminted')).rejects.toThrow(
      new RegExp(`^${connectErrorMessage}`)
    )
  })
})
