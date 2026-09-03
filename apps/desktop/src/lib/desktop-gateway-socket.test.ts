import { afterEach, describe, expect, it, vi } from 'vitest'

import { createDesktopGatewaySocket, DesktopMainGatewaySocket, type GatewayWsBridge } from './desktop-gateway-socket'
import { shouldDialGatewayFromMain } from './gateway-ws-target'

describe('shouldDialGatewayFromMain', () => {
  it('is true for remote wss /api/ws and false for loopback', () => {
    expect(shouldDialGatewayFromMain('wss://gw.example.com/api/ws?ticket=t')).toBe(true)
    expect(shouldDialGatewayFromMain('ws://127.0.0.1:52515/api/ws?token=t')).toBe(false)
  })
})

describe('DesktopMainGatewaySocket', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('queues events that arrive before open() resolves the id', async () => {
    const listeners = new Set<(event: any) => void>()
    const bridge: GatewayWsBridge = {
      close: vi.fn(),
      open: vi.fn().mockImplementation(() => new Promise(resolve => {
        queueMicrotask(() => {
          for (const listener of listeners) {
            listener({ id: 'sock-1', type: 'open' })
          }

          resolve({ ok: true, id: 'sock-1' })
        })
      })),
      send: vi.fn(),
      subscribe: callback => {
        listeners.add(callback)

        return () => listeners.delete(callback)
      }
    }

    const opened = vi.fn()
    const socket = new DesktopMainGatewaySocket('wss://gw.example.com/api/ws?ticket=t', bridge)
    socket.addEventListener('open', opened, { once: true })

    await vi.waitFor(() => {
      expect(opened).toHaveBeenCalledOnce()
    })
    expect(socket.readyState).toBe(1)
  })

  it('falls back to Chromium WebSocket for loopback URLs', () => {
    class FakeWs {
      url: string
      constructor(url: string) {
        this.url = url
      }
    }

    vi.stubGlobal('WebSocket', FakeWs)

    const socket = createDesktopGatewaySocket('ws://127.0.0.1:9/api/ws?token=t')

    expect(socket).toBeInstanceOf(FakeWs)
    vi.unstubAllGlobals()
  })
})
