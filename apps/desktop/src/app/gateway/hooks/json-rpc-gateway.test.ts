import { GatewayRequestError, isGatewayPreDispatchError, JsonRpcGatewayClient, type WebSocketLike } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

// ── Transport-boundary dispatch authority ─────────────────────────────────
// The gateway client must tag every failure with whether the request frame was
// written to the socket (dispatched). The composer's draft-restore decision on
// prompt.submit and the reconnect wrapper's replay decision both key off it:
// a pre-dispatch failure is CERTAIN (nothing was sent — restore/replay safe),
// a post-dispatch no-response failure is UNCERTAIN (the gateway may have
// accepted the request — never restore the draft, never replay).

class FakeSocket {
  readyState = WebSocket.OPEN
  sent: string[] = []
  sendShouldThrow: null | Error = null
  private listeners = new Map<string, Set<(event: unknown) => void>>()

  addEventListener(type: string, handler: (event: unknown) => void): void {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set())
    }

    this.listeners.get(type)!.add(handler)

    // The client's connect() awaits an 'open' event; fire it on a microtask
    // once the listener is attached.
    if (type === 'open') {
      queueMicrotask(() => handler({}))
    }
  }

  removeEventListener(type: string, handler: (event: unknown) => void): void {
    this.listeners.get(type)?.delete(handler)
  }

  send(data: string): void {
    if (this.sendShouldThrow) {
      throw this.sendShouldThrow
    }

    this.sent.push(data)
  }

  close(): void {}

  emit(type: string, event: unknown): void {
    for (const handler of [...(this.listeners.get(type) ?? [])]) {
      handler(event)
    }
  }
}

function makeClient(socket: FakeSocket, requestTimeoutMs = 50): JsonRpcGatewayClient {
  return new JsonRpcGatewayClient({
    requestTimeoutMs,
    socketFactory: () => socket as unknown as WebSocketLike
  })
}

async function openClient(socket: FakeSocket, requestTimeoutMs?: number): Promise<JsonRpcGatewayClient> {
  const client = makeClient(socket, requestTimeoutMs)

  await client.connect('ws://gateway')

  return client
}

async function rejectOf<T>(promise: Promise<T>): Promise<GatewayRequestError> {
  try {
    await promise
  } catch (error) {
    return error as GatewayRequestError
  }

  throw new Error('expected the request to reject')
}

describe('JsonRpcGatewayClient dispatch authority', () => {
  it("rejects with a PRE-dispatch typed error when no socket is open — dispatched:false (the request never left the machine)", async () => {
    const client = new JsonRpcGatewayClient()

    const error = await rejectOf(client.request('prompt.submit', { text: 'x' }))

    expect(error).toBeInstanceOf(GatewayRequestError)
    expect(error.kind).toBe('not_connected')
    expect(error.dispatched).toBe(false)
    expect(error.message).toBe('gateway not connected')
    expect(isGatewayPreDispatchError(error)).toBe(true)
  })

  it("rejects with a POST-dispatch typed error when the response times out — dispatched:true (the gateway may have accepted it)", async () => {
    const socket = new FakeSocket()
    const client = await openClient(socket, 25)

    const error = await rejectOf(client.request('prompt.submit', { text: 'x' }))

    expect(error).toBeInstanceOf(GatewayRequestError)
    expect(error.kind).toBe('timeout')
    expect(error.dispatched).toBe(true)
    expect(error.message).toMatch(/request timed out after \d+s: prompt\.submit/)
    expect(isGatewayPreDispatchError(error)).toBe(false)
    // The frame did leave the machine before the timeout fired.
    expect(socket.sent).toHaveLength(1)
  })

  it("rejects pending calls with a POST-dispatch typed error when the socket closes — dispatched:true", async () => {
    const socket = new FakeSocket()
    const client = await openClient(socket)

    const pending = client.request('prompt.submit', { text: 'x' })
    socket.emit('close', {})

    const error = await rejectOf(pending)

    expect(error).toBeInstanceOf(GatewayRequestError)
    expect(error.kind).toBe('closed')
    expect(error.dispatched).toBe(true)
    expect(error.message).toBe('WebSocket closed')
    expect(isGatewayPreDispatchError(error)).toBe(false)
  })

  it("rejects with a PRE-dispatch typed error when socket.send throws — dispatched:false", async () => {
    const socket = new FakeSocket()
    socket.sendShouldThrow = new Error('socket is dead')
    const client = await openClient(socket)

    const error = await rejectOf(client.request('prompt.submit', { text: 'x' }))

    expect(error).toBeInstanceOf(GatewayRequestError)
    expect(error.kind).toBe('send_failed')
    expect(error.dispatched).toBe(false)
    expect(error.message).toBe('socket is dead')
    expect(isGatewayPreDispatchError(error)).toBe(true)
  })

  it("rejects with an RPC-error typed error when the server answers with an error frame — a certain rejection", async () => {
    const socket = new FakeSocket()
    const client = await openClient(socket)

    const pending = client.request('prompt.submit', { text: 'x' })
    socket.emit('message', {
      data: JSON.stringify({ jsonrpc: '2.0', id: 'r1', error: { message: '4009 session busy' } })
    })

    const error = await rejectOf(pending)

    expect(error).toBeInstanceOf(GatewayRequestError)
    expect(error.kind).toBe('rpc')
    expect(error.dispatched).toBe(true)
    expect(error.message).toBe('4009 session busy')
    // A server answer is NOT an uncertain outcome — the request was processed.
    expect(isGatewayPreDispatchError(error)).toBe(false)
  })

  it('still resolves normally when the response frame arrives', async () => {
    const socket = new FakeSocket()
    const client = await openClient(socket)

    const pending = client.request('session.status', {}).catch(e => e)
    socket.emit('message', {
      data: JSON.stringify({ jsonrpc: '2.0', id: 'r1', result: { output: 'idle' } })
    })

    await expect(pending).resolves.toEqual({ output: 'idle' })
  })
})

describe('isGatewayPreDispatchError', () => {
  it('keys off the typed dispatched flag when available', () => {
    expect(isGatewayPreDispatchError(new GatewayRequestError('not_connected', 'x', false))).toBe(true)
    expect(isGatewayPreDispatchError(new GatewayRequestError('send_failed', 'x', false))).toBe(true)
    expect(isGatewayPreDispatchError(new GatewayRequestError('closed', 'x', true))).toBe(false)
    expect(isGatewayPreDispatchError(new GatewayRequestError('timeout', 'x', true))).toBe(false)
    expect(isGatewayPreDispatchError(new GatewayRequestError('rpc', 'x', true))).toBe(false)
  })

  it('falls back to message shape for untyped errors (other bridges)', () => {
    expect(isGatewayPreDispatchError(new Error('Hermes gateway is not connected'))).toBe(true)
    expect(isGatewayPreDispatchError(new Error('Hermes gateway connection closed'))).toBe(false)
    // A message that hints at both is treated conservatively — may have been sent.
    expect(isGatewayPreDispatchError(new Error('connection closed: not connected'))).toBe(false)
    expect(isGatewayPreDispatchError(new Error('request timed out after 30s: prompt.submit'))).toBe(false)
    expect(isGatewayPreDispatchError('not connected')).toBe(true)
  })
})
