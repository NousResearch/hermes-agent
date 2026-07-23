import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  pluginRest,
  pluginSocket,
  setApiRequestProfile
} from './hermes'

describe('Desktop plugin IPC transport', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn().mockResolvedValue({ ok: true })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        api,
        getPluginWsUrl: vi
          .fn()
          .mockResolvedValueOnce('ws://127.0.0.1/api/plugins/kanban/events?ticket=one')
          .mockResolvedValueOnce('ws://127.0.0.1/api/plugins/kanban/events?ticket=two')
      }
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('scopes JSON plugin REST under the manifest id and rejects traversal', async () => {
    await pluginRest('kanban', '/board')
    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/plugins/kanban/board' }))
    await expect(pluginRest('kanban', '/../other')).rejects.toThrow(/Unsafe/)
    await expect(pluginRest('../kanban', '/board')).rejects.toThrow(/manifest id/)
  })

  it('passes the active profile through plugin REST requests', async () => {
    setApiRequestProfile('kanban-profile')
    await pluginRest('kanban', '/tasks')

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ profile: 'kanban-profile' }))
    setApiRequestProfile(null)
  })

  it('mints a fresh plugin WebSocket URL for reconnects', () => {
    const sockets: Array<{
      close: () => void
      onclose?: () => void
      onmessage?: (event: MessageEvent) => void
      url: string
    }> = []

    const OriginalWebSocket = window.WebSocket
    let closeCallCount = 0

    vi.useFakeTimers()
    vi.stubGlobal(
      'WebSocket',
      class {
        close: () => void
        onclose: (() => void) | undefined
        onmessage: ((event: MessageEvent) => void) | undefined
        url: string

        constructor(url: string) {
          closeCallCount += 1
          this.url = url
          this.close = vi.fn()
          sockets.push(this)
        }
      }
    )

    const dispose = pluginSocket('kanban', '/events', vi.fn())

    return Promise.resolve()
      .then(async () => {
        await Promise.resolve()
        expect(sockets).toHaveLength(1)
        expect(sockets[0].url).toContain('ticket=one')
        sockets[0].onclose?.()
        expect(closeCallCount).toBe(1)
        await vi.runOnlyPendingTimersAsync()
        expect(sockets).toHaveLength(2)
        expect(sockets[1].url).toContain('ticket=two')
      })
      .finally(() => {
        dispose()
        vi.useRealTimers()
        vi.stubGlobal('WebSocket', OriginalWebSocket)
      })
  })
})
