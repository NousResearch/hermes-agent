import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'

type RelayWindow = Window & {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_BASE_PATH__?: string
  __HERMES_GATEWAY_RELAY__?: boolean
  hermesDesktop?: Window['hermesDesktop']
}

const win = window as unknown as RelayWindow

const MINI = { added_at: 1, id: 'mini', label: 'Mini', scopes: ['files'], url: 'https://mini.example.ts.net' }

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' }, status })

/** The serving gateway at /hermes, relaying to Mini; answers by path and records every request. */
function servingGateway() {
  const requests: { method: string; path: string; search: string }[] = []

  const fetchMock = vi.fn(async (url: URL, init?: RequestInit) => {
    const path = url.pathname
    requests.push({ method: init?.method || 'GET', path, search: url.search })

    if (path === '/hermes/api/webapp/relay/gateways') {
      return json({ gateways: [MINI] })
    }

    // Mini is down: it cannot confirm it revoked the grant, so only forgetting it here removes it.
    if (path === '/hermes/api/webapp/relay/gateways/mini' && init?.method === 'DELETE') {
      return json(
        url.searchParams.get('forget') === '1'
          ? { removed: true, revoked: false }
          : { detail: "Mini did not confirm it revoked this gateway's access: it answered 503.", removed: false }
      )
    }

    if (path === '/hermes/api/relay/mini/api/profiles') {
      return json({ profiles: [{ name: 'default' }, { name: 'research' }] })
    }

    if (path === '/hermes/api/relay/mini/api/profiles/sessions') {
      return json({ sessions: [{ id: 'm1', profile: 'research', title: 'On Mini' }], total: 1 })
    }

    if (path === '/hermes/api/profiles') {
      return json({ profiles: [{ name: 'default' }] })
    }

    if (path.endsWith('/api/webapp/socket-ticket')) {
      const socket = url.searchParams.get('socket')
      const where = path.startsWith('/hermes/api/relay/mini/') ? 'mini' : 'hub'

      return json({ ticket: `${where}-${socket}`, ttl_seconds: 30 })
    }

    return json({})
  })

  vi.stubGlobal('fetch', fetchMock)

  return requests
}

function install({ relay = true } = {}) {
  win.__HERMES_AUTH_REQUIRED__ = true
  win.__HERMES_BASE_PATH__ = '/hermes'

  if (relay) {
    win.__HERMES_GATEWAY_RELAY__ = true
  }

  expect(installBrowserDesktopBridge()).toBe(true)

  return win.hermesDesktop!
}

afterEach(() => {
  delete win.__HERMES_AUTH_REQUIRED__
  delete win.__HERMES_BASE_PATH__
  delete win.__HERMES_GATEWAY_RELAY__
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  window.localStorage.clear()
  $connection.set(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('a Webapp whose gateway relays to others', () => {
  it('lists the relayed gateways beside its own as registry connections', async () => {
    servingGateway()
    const registry = await install().connections!.list()

    expect(registry.connections.map(connection => [connection.id, connection.kind])).toEqual([
      ['local', 'local'],
      ['mini', 'remote']
    ])
    expect(registry.primary).toBe('local')
  })

  it('keeps a gateway that did not confirm the revocation until it is removed here only', async () => {
    const requests = servingGateway()
    const desktop = install()
    await desktop.connections!.list()
    const events: string[] = []
    desktop.connections!.onChanged!(event => events.push(`${event.connectionId}:${event.reason}`))

    await expect(desktop.connections!.remove('mini')).rejects.toThrow('Mini did not confirm')
    expect(events).toEqual([])

    await expect(desktop.gatewayRelay!.removeGateway('mini', { forget: true })).resolves.toEqual({
      removed: true,
      revoked: false
    })
    expect(requests.filter(request => request.method === 'DELETE').map(request => request.search)).toEqual([
      '',
      '?forget=1'
    ])
    expect(events).toEqual(['mini:removed'])
  })

  it("sends a relayed connection's requests through the relay and owns their rows to it", async () => {
    const requests = servingGateway()
    const desktop = install()
    await desktop.connections!.list()

    const listed = await desktop.api<{ sessions: { connection_id?: string }[] }>({
      connectionId: 'mini',
      path: '/api/profiles/sessions'
    })

    expect(requests.at(-1)?.path).toBe('/hermes/api/relay/mini/api/profiles/sessions')
    expect(listed.sessions.map(session => session.connection_id)).toEqual(['mini'])
  })

  it("dials a relayed gateway's socket with its ticket and the serving gateway's relay ticket", async () => {
    servingGateway()
    const desktop = install()
    await desktop.connections!.list()

    const dial = (await desktop.getGatewayWsUrlFor!({ connectionId: 'mini', profile: 'research' })) as {
      ok: boolean
      wsUrl: string
    }

    expect(dial.ok).toBe(true)
    const url = new URL(dial.wsUrl)
    expect(url.pathname).toBe('/hermes/api/relay/mini/api/ws')
    expect(url.searchParams.get('ticket')).toBe('mini-gateway')
    expect(url.searchParams.get('relay_ticket')).toBe('hub-relay')
    expect(url.searchParams.get('profile')).toBe('research')

    // The descriptor carries no credential; each dial mints its own pair.
    const connection = await desktop.getConnectionFor!({ connectionId: 'mini', profile: 'research' })
    expect(connection.connectionId).toBe('mini')
    expect(new URL(connection.wsUrl).searchParams.has('ticket')).toBe(false)
  })

  it('follows the window onto a relayed gateway for its host terminal and files', async () => {
    servingGateway()
    const desktop = install()
    await desktop.connections!.list()
    $connection.set({ ...(await desktop.getConnectionFor!({ connectionId: 'mini' })) })

    const dialed: URL[] = []

    vi.stubGlobal(
      'WebSocket',
      class {
        static OPEN = 1
        readyState = 0
        onclose: ((event: CloseEvent) => void) | null = null
        onerror: (() => void) | null = null
        onmessage: ((event: MessageEvent) => void) | null = null
        onopen: (() => void) | null = null
        constructor(url: string | URL) {
          dialed.push(new URL(url))
          queueMicrotask(() => this.onclose?.(new CloseEvent('close', { code: 4401 })))
        }
        close() {}
        send() {}
      }
    )

    await desktop.terminal.start({ restoreKey: 'relayed' }).catch(() => undefined)
    expect(dialed[0]?.pathname).toBe('/hermes/api/relay/mini/api/host-terminal')
    expect(dialed[0]?.searchParams.get('ticket')).toBe('mini-host-terminal')
    expect(dialed[0]?.searchParams.get('relay_ticket')).toBe('hub-relay')

    const stream = new URL(await desktop.getGatewayFileStreamUrl!({ connectionId: 'mini', path: '/tmp/a.png' }))
    expect(stream.pathname.startsWith('/hermes/api/relay/mini/')).toBe(true)
  })

  it('lists the profiles of every relayed gateway in the agent roster', async () => {
    servingGateway()
    const roster = await install().getAgentRoster!()

    expect(roster.agents.filter(agent => agent.connectionId === 'mini').map(agent => agent.profile)).toEqual([
      'default',
      'research'
    ])
    expect(roster.sources.map(source => [source.connectionId, source.reachable])).toEqual([
      ['local', true],
      ['mini', true]
    ])
  })

  it('reaches only its own gateway when the relay is off', async () => {
    const requests = servingGateway()
    const desktop = install({ relay: false })

    expect(desktop.gatewayRelay).toBeUndefined()
    await expect(desktop.getConnectionFor!({ connectionId: 'mini' })).rejects.toThrow(/mini/)
    expect(requests.some(request => request.path.includes('/relay/'))).toBe(false)
  })
})
