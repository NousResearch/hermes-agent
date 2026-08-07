// connect() must reject before WebSocket coerces garbage into
// `ws://<origin>/[object%20Object]` (#68250 stale-emit boot loop).

import { JsonRpcGatewayClient } from '@hermes/shared'
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

class PendingSocket {
  static OPEN = 1
  readyState = 0
  private listeners = new Map<string, Set<() => void>>()

  addEventListener = vi.fn((type: string, handler: () => void) => {
    const handlers = this.listeners.get(type) ?? new Set()
    handlers.add(handler)
    this.listeners.set(type, handlers)
  })

  removeEventListener = vi.fn((type: string, handler: () => void) => {
    this.listeners.get(type)?.delete(handler)
  })

  close = vi.fn(() => {
    for (const handler of this.listeners.get('close') ?? []) {
      handler()
    }
  })

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

  it('settles an in-flight connect immediately when closed', async () => {
    vi.stubGlobal('WebSocket', PendingSocket)

    const socket = new PendingSocket()

    const client = new JsonRpcGatewayClient({
      connectTimeoutMs: 15_000,
      socketFactory: () => socket as unknown as WebSocket
    })

    const connecting = client.connect('ws://127.0.0.1:1234/api/ws')
    client.close()

    await expect(connecting).rejects.toThrow('WebSocket closed')
    expect(client.connectionState).toBe('closed')
  })
})
