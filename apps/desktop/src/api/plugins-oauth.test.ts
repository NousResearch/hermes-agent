import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import { pluginSocket } from './plugins'

// OAuth cookie-auth remotes historically never dialed the plugin event socket:
// pluginSocket bailed on authMode === 'oauth' (single-use WS tickets were
// minted only for the gateway socket), so the kanban board fell back to its
// 60s poll and completion notifications never fired (#117541). The dial now
// goes through the main-process getPluginWsUrl door (mint + URL build live in
// main; the renderer only opens what it gets back).
class FakeWebSocket {
  static opened: FakeWebSocket[] = []
  static lastUrl = ''
  onclose: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  url: string
  constructor(url: string) {
    this.url = url
    FakeWebSocket.lastUrl = url
    FakeWebSocket.opened.push(this)
  }
  close() {}
}

describe('pluginSocket OAuth ticket dial (#117541)', () => {
  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.restoreAllMocks()
  })

  function installBridge(over: Record<string, unknown> = {}) {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        getConnection: vi.fn(async () => ({
          authMode: 'oauth',
          baseUrl: 'https://remote.invalid',
          mode: 'remote',
          token: '',
          wsUrl: 'wss://remote.invalid/api/ws'
        })),
        getPluginWsUrl: vi.fn(async () => ({ ok: true, wsUrl: 'wss://remote.invalid/api/plugins/kanban/events?ticket=t' })),
        ...over
      }
    })
  }

  it('dials the minted plugin URL on an OAuth remote', async () => {
    vi.stubGlobal('WebSocket', FakeWebSocket)
    installBridge()

    const dispose = pluginSocket('kanban', '/events', () => {})

    await vi.waitFor(() => expect(FakeWebSocket.opened).toHaveLength(1))
    expect(FakeWebSocket.lastUrl).toContain('/api/plugins/kanban/events?ticket=')

    dispose()
    vi.unstubAllGlobals()
  })

  it('keeps the polling fallback when the mint is rejected', async () => {
    vi.stubGlobal('WebSocket', FakeWebSocket)
    FakeWebSocket.opened = []
    installBridge({ getPluginWsUrl: vi.fn(async () => ({ ok: false, error: 'expired' })) })

    const dispose = pluginSocket('kanban', '/events', () => {})
    await new Promise(resolve => setTimeout(resolve, 50))

    expect(FakeWebSocket.opened).toHaveLength(0)

    dispose()
    vi.unstubAllGlobals()
  })

  it('scopes the mint request to the active connection and profile', async () => {
    vi.stubGlobal('WebSocket', FakeWebSocket)
    FakeWebSocket.opened = []
    setApiRequestConnection('homelab')
    setApiRequestProfile('research')
    const getPluginWsUrl = vi.fn(async () => ({ ok: true, wsUrl: 'wss://homelab.invalid/api/plugins/kanban/events?ticket=t' }))
    installBridge({ getPluginWsUrl })

    const dispose = pluginSocket('kanban', '/events', () => {})
    await vi.waitFor(() => expect(FakeWebSocket.opened).toHaveLength(1))

    expect(getPluginWsUrl).toHaveBeenCalledWith(
      { connectionId: 'homelab', profile: 'research' },
      '/api/plugins/kanban/events'
    )

    dispose()
    vi.unstubAllGlobals()
  })
})
