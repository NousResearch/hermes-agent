// connect() must reject before WebSocket coerces garbage into
// `ws://<origin>/[object%20Object]` (#68250 stale-emit boot loop).

import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { HermesGateway } from '@/hermes'

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

class SlowFakeSocket {
  static OPEN = 1
  readyState = 0
  addEventListener = vi.fn((type: string, handler: () => void) => {
    if (type === 'open') {
      setTimeout(() => {
        this.readyState = SlowFakeSocket.OPEN
        handler()
      }, 20_000)
    }
  })
  removeEventListener = vi.fn()
  close = vi.fn()
  send = vi.fn()
}

class NeverOpenSocket {
  static OPEN = 1
  readyState = 0
  addEventListener = vi.fn()
  removeEventListener = vi.fn()
  close = vi.fn()
  send = vi.fn()
}

describe('JsonRpcGatewayClient connect()', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', FakeSocket) // jsdom has none; class reads WebSocket.OPEN
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
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

  it('allows Desktop to survive a slow local backend boot', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', SlowFakeSocket)

    const client = new HermesGateway()
    const connected = client.connect('ws://127.0.0.1:1234/api/ws?token=t')

    await vi.advanceTimersByTimeAsync(20_000)
    await expect(connected).resolves.toBeUndefined()
    expect(client.connectionState).toBe('open')
  })

  it('keeps the shared gateway client connect timeout at 15 seconds', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', SlowFakeSocket)

    const client = new JsonRpcGatewayClient()
    const connected = expect(client.connect('ws://127.0.0.1:1234/api/ws?token=t')).rejects.toThrow(
      'WebSocket connection failed'
    )

    await vi.advanceTimersByTimeAsync(15_000)
    await connected
    expect(client.connectionState).toBe('error')
  })

  it('still fails a Desktop dial that never opens within 60 seconds', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', NeverOpenSocket)

    const client = new HermesGateway()
    const connected = client.connect('ws://127.0.0.1:1234/api/ws?token=t').then(
      () => null,
      error => error
    )

    await vi.advanceTimersByTimeAsync(60_000)
    await expect(connected).resolves.toEqual(new Error('Could not connect to Hermes gateway'))
    expect(client.connectionState).toBe('error')
  })
})
