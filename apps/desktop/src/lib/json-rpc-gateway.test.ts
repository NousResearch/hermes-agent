import { afterEach, describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayClient } from '@hermes/shared'

type Listener = () => void

class FakeSocket {
  readonly sent: string[] = []
  readyState = 1
  private listeners = new Map<string, Set<Listener>>()

  addEventListener(type: string, listener: Listener): void {
    const listeners = this.listeners.get(type) ?? new Set<Listener>()

    listeners.add(listener)
    this.listeners.set(type, listeners)
  }

  close(): void {
    this.readyState = 3
    this.emit('close')
  }

  emit(type: string): void {
    for (const listener of this.listeners.get(type) ?? []) {
      listener()
    }
  }

  removeEventListener(type: string, listener: Listener): void {
    this.listeners.get(type)?.delete(listener)
  }

  send(payload: string): void {
    this.sent.push(payload)
  }
}

describe('JsonRpcGatewayClient', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('closes a stale socket when an RPC request times out', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', { OPEN: 1 })

    const socket = new FakeSocket()
    const states: string[] = []
    const client = new JsonRpcGatewayClient({
      createRequestId: nextId => nextId,
      requestTimeoutMs: 100,
      socketFactory: () => socket as unknown as WebSocket
    })

    client.onState(state => states.push(state))

    const connect = client.connect('ws://localhost/ws')
    socket.emit('open')
    await connect

    const request = client
      .request('prompt.submit', { session_id: 'sid', text: 'hello' })
      .then(
        () => null,
        error => error as Error
      )

    await vi.advanceTimersByTimeAsync(100)

    await expect(request).resolves.toMatchObject({ message: 'request timed out: prompt.submit' })
    expect(socket.readyState).toBe(3)
    expect(client.connectionState).toBe('closed')
    expect(states.at(-1)).toBe('closed')
  })
})
