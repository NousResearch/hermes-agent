import { describe, expect, it } from 'vitest'

import { JsonRpcGatewayClient } from './json-rpc-gateway'

class FakeSocket {
  readyState = 0
  private listeners = new Map<string, Set<(...args: unknown[]) => void>>()

  addEventListener(type: string, handler: (...args: unknown[]) => void) {
    const set = this.listeners.get(type) ?? new Set()
    set.add(handler)
    this.listeners.set(type, set)
  }

  removeEventListener(type: string, handler: (...args: unknown[]) => void) {
    this.listeners.get(type)?.delete(handler)
  }

  close() {
    this.readyState = 3
    this.emit('close')
  }

  openNow() {
    this.readyState = 1
    this.emit('open')
  }

  private emit(type: string) {
    for (const handler of this.listeners.get(type) ?? []) {
      handler()
    }
  }
}

describe('JsonRpcGatewayClient.connect', () => {
  it('shares an in-flight connect promise instead of short-circuiting', async () => {
    let socket: FakeSocket | null = null

    const client = new JsonRpcGatewayClient({
      connectTimeoutMs: 5_000,
      socketFactory: () => {
        socket = new FakeSocket()

        return socket as unknown as WebSocket
      }
    })

    const first = client.connect('ws://127.0.0.1:9/api/ws')
    const second = client.connect('ws://127.0.0.1:9/api/ws')

    expect(client.connectionState).toBe('connecting')
    expect(socket).not.toBeNull()

    queueMicrotask(() => socket?.openNow())

    await Promise.all([first, second])
    expect(client.connectionState).toBe('open')
  })
})
