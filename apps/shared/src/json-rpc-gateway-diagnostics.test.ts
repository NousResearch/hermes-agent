import { beforeEach, describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayClient } from './json-rpc-gateway'

/**
 * Node's vitest environment has no CloseEvent global (the existing
 * json-rpc-gateway-replay.test.ts relies on it and is not wired into CI).
 * Polyfill the minimal shape the gateway's close handler reads.
 */
class FakeCloseEvent extends Event {
  code = 1006
  reason = ''
  wasClean = false
}

if (typeof globalThis.CloseEvent === 'undefined') {
  ;(globalThis as Record<string, unknown>).CloseEvent = FakeCloseEvent
}

class FakeWebSocket extends EventTarget {
  static OPEN = 1
  static instances: FakeWebSocket[] = []

  readyState = 0
  sent: string[] = []
  url: string

  constructor(url: string) {
    super()
    this.url = url
    FakeWebSocket.instances.push(this)
  }

  send(data: string): void {
    this.sent.push(data)
  }

  close(): void {
    this.readyState = 3
    this.dispatchEvent(new CloseEvent('close'))
  }

  open(): void {
    this.readyState = 1
    this.dispatchEvent(new Event('open'))
  }

  serverFrame(obj: unknown): void {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(obj) }))
  }
}

let sockets: FakeWebSocket[]

const makeClient = () => {
  const client = new JsonRpcGatewayClient({
    socketFactory: url => new FakeWebSocket(url) as unknown as WebSocket,
    heartbeatIntervalMs: 0,
    heartbeatDeadlineMs: 0,
    connectTimeoutMs: 1000
  })

  return client
}

describe('JsonRpcGatewayClient WS disconnect diagnostics', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
    sockets = FakeWebSocket.instances as unknown as FakeWebSocket[]
  })

  it('logs the socket close event with code and reason', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const client = makeClient()
    const p = client.connect('ws://x')
    sockets[0].open()
    await p

    sockets[0].close()

    const log = errorSpy.mock.calls.map(c => c.join(' ')).join('\n')
    expect(log).toContain('[gateway] socket close event')
    expect(log).toContain('code=1006')
    expect(log).toContain('reason=')

    errorSpy.mockRestore()
    client.close()
  })

  it('logs close() called with the current state', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const client = makeClient()
    const p = client.connect('ws://x')
    sockets[0].open()
    await p

    client.close()

    const log = errorSpy.mock.calls.map(c => c.join(' ')).join('\n')
    expect(log).toContain('[gateway] close() called')
    expect(log).toContain('state=')

    errorSpy.mockRestore()
  })

  it('logs invalidateSocket with the error message and lastInboundAgeMs', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const client = makeClient()
    const p = client.connect('ws://x')
    sockets[0].open()
    await p

    client.invalidate('drop')

    const log = errorSpy.mock.calls.map(c => c.join(' ')).join('\n')
    expect(log).toContain('[gateway] invalidateSocket:')
    expect(log).toContain('drop')
    expect(log).toContain('lastInboundAgeMs=')

    errorSpy.mockRestore()
    client.close()
  })
})
