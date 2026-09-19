import { describe, expect, it } from 'vitest'

import { JsonRpcGatewayClient } from './json-rpc-gateway'

/** EventTarget-based WebSocket stand-in that never opens, so each failure leg can be driven by hand. */
class StuckSocket extends EventTarget {
  static OPEN = 1
  static last: StuckSocket | null = null

  readyState = 0

  constructor() {
    super()
    StuckSocket.last = this
  }

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
  const client = new JsonRpcGatewayClient({
    socketFactory: () => new StuckSocket() as unknown as WebSocket,
    heartbeatIntervalMs: 0,
    heartbeatDeadlineMs: 0,
    connectTimeoutMs,
    connectErrorMessage
  })

  const pending = client.connect('ws://gateway.test/api/ws')
  pending.catch(() => {})

  return { client, pending, socket: StuckSocket.last as StuckSocket }
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

  it('an error before open still names the base message so existing "sidecar down" matchers keep working', async () => {
    const { pending, socket } = dial()
    socket.dispatchEvent(new Event('error'))
    const error = await rejection(pending)

    expect(error.message.startsWith(connectErrorMessage)).toBe(true)
    expect(error.message).toContain('error before open')
  })
})
