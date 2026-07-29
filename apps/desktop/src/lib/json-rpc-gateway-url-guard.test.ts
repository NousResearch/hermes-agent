// connect() must reject before WebSocket coerces garbage into
// `ws://<origin>/[object%20Object]` (#68250 stale-emit boot loop).

import { isGatewayReauthRequired, JsonRpcGatewayClient } from '@hermes/shared'
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

// Full listener bookkeeping (unlike FakeSocket above, which only special-cases
// 'open') so a test can fire 'close' with an arbitrary code before 'open' ever
// happens — the gateway accepting the WS upgrade and immediately closing with
// an app-level auth code (hermes_cli/web_server.py) instead of firing 'error'.
class FakeCloseCodeSocket {
  static OPEN = 1
  readyState = 0
  private listeners: Record<string, Set<(event?: { code?: number }) => void>> = {}

  addEventListener(type: string, handler: (event?: { code?: number }) => void) {
    ;(this.listeners[type] ??= new Set()).add(handler)
  }

  removeEventListener(type: string, handler: (event?: { code?: number }) => void) {
    this.listeners[type]?.delete(handler)
  }

  close = vi.fn()
  send = vi.fn()

  emitClose(code: number) {
    for (const handler of this.listeners.close ?? []) {
      handler({ code })
    }
  }
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

  it('rejects a handshake closed with an auth code (4401) as reauth-required', async () => {
    let socket: FakeCloseCodeSocket | undefined
    const client = new JsonRpcGatewayClient({
      socketFactory: () => {
        socket = new FakeCloseCodeSocket()

        return socket as unknown as WebSocket
      }
    })

    const connecting = client.connect('ws://127.0.0.1:1234/api/ws?token=stale')

    // 'open' never fires — this simulates the gateway accepting the upgrade
    // and immediately closing with an app-level auth code instead.
    socket?.emitClose(4401)

    const error = await connecting.catch(e => e)
    expect(isGatewayReauthRequired(error)).toBe(true)
    expect(client.connectionState).toBe('closed')
  })

  it('rejects a handshake closed with a non-auth code as a plain (retryable) error', async () => {
    let socket: FakeCloseCodeSocket | undefined
    const client = new JsonRpcGatewayClient({
      socketFactory: () => {
        socket = new FakeCloseCodeSocket()

        return socket as unknown as WebSocket
      }
    })

    const connecting = client.connect('ws://127.0.0.1:1234/api/ws?token=t')

    socket?.emitClose(1006)

    const error = await connecting.catch(e => e)
    expect(isGatewayReauthRequired(error)).toBe(false)
  })

  it('rejects a handshake closed with a policy code (4403) without tagging it as reauth-required', async () => {
    // 4403 is host/origin/policy rejection (embedded chat disabled, wrong Host
    // header, etc. — web_server.py:18632,18640), not an expired session.
    // Signing in again can't fix it, so — unlike 4401 — it must NOT set
    // needsOauthLogin; the desktop boot-level assertion on the error string
    // alone can't tell these two apart, since both fail fast either way.
    let socket: FakeCloseCodeSocket | undefined
    const client = new JsonRpcGatewayClient({
      socketFactory: () => {
        socket = new FakeCloseCodeSocket()

        return socket as unknown as WebSocket
      }
    })

    const connecting = client.connect('ws://127.0.0.1:1234/api/ws?token=t')

    socket?.emitClose(4403)

    const error = await connecting.catch(e => e)
    expect(isGatewayReauthRequired(error)).toBe(false)
    expect((error as { wsCloseCode?: number }).wsCloseCode).toBe(4403)
  })

  it('close() during an in-flight handshake rejects it immediately, and the abandoned connect timeout never corrupts a later open', async () => {
    vi.useFakeTimers()

    try {
      let firstSocket: FakeCloseCodeSocket | undefined
      let dialCount = 0
      const client = new JsonRpcGatewayClient({
        socketFactory: () => {
          dialCount += 1

          if (dialCount === 1) {
            firstSocket = new FakeCloseCodeSocket()

            return firstSocket as unknown as WebSocket
          }

          // Second dial: a plain socket that opens on the next tick, like the
          // 'accepts ws:// and wss://' case above.
          return new FakeSocket() as unknown as WebSocket
        }
      })

      const firstConnect = client.connect('ws://127.0.0.1:1234/api/ws?token=t')

      // Caller tears the client down while connect() hasn't settled yet
      // (firstSocket never fires 'open' — close = vi.fn(), a silent no-op, so
      // nothing but close() itself can settle this handshake).
      client.close()

      const closeError = await firstConnect.catch(e => e)
      expect(closeError).toMatchObject({ message: 'WebSocket closed' })
      expect(client.connectionState).toBe('closed')

      // A fresh connect() opens normally.
      const secondConnect = client.connect('ws://127.0.0.1:1234/api/ws?token=t2')
      await vi.advanceTimersByTimeAsync(0)
      await secondConnect
      expect(client.connectionState).toBe('open')

      // The FIRST connect()'s abandoned 15s connect-timeout timer must not
      // still be armed: if it fired here it would stomp this back to 'error'
      // even though a completely different (newer) socket is now open.
      await vi.advanceTimersByTimeAsync(15_000)
      expect(client.connectionState).toBe('open')
    } finally {
      vi.useRealTimers()
    }
  })
})
