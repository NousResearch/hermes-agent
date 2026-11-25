import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry, DesktopRegistryConnection } from '@/global'
import { $connection } from '@/store/session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { NoHostedAgentError } from './browser-saved-connections'

/**
 * A host that runs no agent of its own (`__HERMES_HOST_AGENT__ === false`): the hosted dashboard. It
 * signs the user in, lists their agents and relays to them; nothing else under /api is its own. The
 * window opens on the list's primary, and the only requests it sends the host itself are the routes
 * below. Everything else goes through `/api/relay/<agent>`.
 */
const HOST_SURFACE = new Set([
  'GET /api/auth/me',
  'GET /api/auth/providers',
  'GET /api/gateway-connections',
  'POST /api/gateway-connections/last-used',
  'POST /api/gateway-connections/launch-mode',
  'POST /api/gateway-connections/alpha/primary',
  'POST /api/gateway-connections/beta/primary',
  'GET /api/ui-preferences',
  'PATCH /api/ui-preferences',
  'POST /api/webapp/socket-ticket?socket=relay'
])

type HostWindow = Window & {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_BASE_PATH__?: string
  __HERMES_HOST_AGENT__?: boolean
  __HERMES_UI_SURFACE__?: string
  hermesDesktop?: Window['hermesDesktop']
}

const win = window as unknown as HostWindow

const agent = (id: string, label: string): DesktopRegistryConnection => ({
  authMode: 'oauth',
  headerNames: [],
  id,
  kind: 'cloud',
  label,
  tokenPreview: null,
  tokenSet: true,
  url: `https://${id}.agents.example`
})

const ALPHA = agent('alpha', 'Alpha')
const BETA = agent('beta', 'Beta')

const registryOf = (rows: DesktopRegistryConnection[], primary = rows[0]?.id ?? ''): DesktopConnectionsRegistry => ({
  connections: rows,
  lastUsed: primary,
  launchMode: 'primary',
  primary,
  secureTokenStorage: true,
  version: 2
})

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' }, status })

const EXPIRED = { detail: 'dashboard session expired', error: 'session_expired', login_url: '/login' }

/** The dashboard host: answers its own routes, relays the rest by agent id, records every request. */
function dashboardHost({
  agents = [ALPHA, BETA],
  primary,
  expired = new Set<string>()
}: { agents?: DesktopRegistryConnection[]; expired?: Set<string>; primary?: string } = {}) {
  const requests: { method: string; path: string; search: string }[] = []

  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: URL, init?: RequestInit) => {
      const method = init?.method || 'GET'
      const path = url.pathname
      requests.push({ method, path, search: url.search })

      if (expired.has(path)) {
        return json(EXPIRED, 401)
      }

      const relayed = path.match(/^\/api\/relay\/([^/]+)(\/.*)$/)

      if (relayed) {
        const [, id, rest] = relayed

        if (rest === '/api/health') {
          return json({ ok: true, version: `${id}-1.0` })
        }

        if (rest === '/api/profiles') {
          return json({ profiles: [{ name: 'default' }] })
        }

        if (rest === '/api/portal') {
          return json({ logged_in: true, portal_url: 'https://portal.example' })
        }

        return json({ from: id, path: rest })
      }

      if (path === '/api/gateway-connections' && method === 'GET') {
        return json(registryOf(agents, primary))
      }

      if (path.startsWith('/api/gateway-connections/') && method === 'POST') {
        return json({ ok: true, registry: registryOf(agents, primary) })
      }

      if (path === '/api/webapp/socket-ticket' && url.searchParams.get('socket') === 'relay') {
        return json({ ticket: 'relay-ticket', ttl_seconds: 30 })
      }

      if (path === '/api/ui-preferences') {
        return json({ exists: false, revision: 0, values: {} })
      }

      return json({ error: 'no_local_agent' }, 404)
    })
  )

  return requests
}

function install({ hostAgent }: { hostAgent?: boolean } = { hostAgent: false }) {
  win.__HERMES_AUTH_REQUIRED__ = true
  win.__HERMES_BASE_PATH__ = ''
  win.__HERMES_UI_SURFACE__ = 'webapp'

  if (hostAgent !== undefined) {
    win.__HERMES_HOST_AGENT__ = hostAgent
  }

  expect(installBrowserDesktopBridge()).toBe(true)

  return win.hermesDesktop!
}

/** The URL a socket dial got; fails the test when the dial was refused. */
function dialed(result: Awaited<ReturnType<Window['hermesDesktop']['getGatewayWsUrl']>>): URL {
  if (typeof result === 'string') {
    return new URL(result)
  }

  if (!result.ok) {
    throw new Error(result.error)
  }

  return new URL(result.wsUrl)
}

/** Requests the host answered itself, as `METHOD /path[?query]`; relayed ones are the agents'. */
const ownRequests = (requests: { method: string; path: string; search: string }[]) =>
  requests
    .filter(request => !request.path.startsWith('/api/relay/'))
    .map(request => `${request.method} ${request.path}${request.search}`)

afterEach(() => {
  delete win.__HERMES_AUTH_REQUIRED__
  delete win.__HERMES_BASE_PATH__
  delete win.__HERMES_HOST_AGENT__
  delete win.__HERMES_UI_SURFACE__
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  window.localStorage.clear()
  $connection.set(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('a Webapp whose host runs no agent', () => {
  it('sends requests boot makes before the window has a connection to the primary agent, never the host', async () => {
    const requests = dashboardHost()
    const desktop = install()

    // The window's connection is not published yet: a request naming none, or `local`, is the primary's.
    const [config, health] = await Promise.all([
      desktop.api<{ from: string; path: string }>({ path: '/api/config' }),
      desktop.api<{ from: string; path: string }>({ connectionId: 'local', path: '/api/profiles/active' })
    ])

    expect(config).toEqual({ from: 'alpha', path: '/api/config' })
    expect(health).toEqual({ from: 'alpha', path: '/api/profiles/active' })
    // Both waited on one read of the list.
    expect(ownRequests(requests)).toEqual(['GET /api/gateway-connections'])
  })

  it("opens the window on the list's primary, its socket relayed with a relay ticket", async () => {
    const requests = dashboardHost({ primary: 'beta' })
    const desktop = install()

    const connection = await desktop.getConnection()
    const socket = await desktop.getGatewayWsUrl()
    const forLocal = await desktop.getGatewayWsUrlFor!({ connectionId: 'local' })

    expect(connection).toMatchObject({ baseUrl: `${window.location.origin}/api/relay/beta`, connectionId: 'beta' })
    expect(dialed(socket).pathname).toBe('/api/relay/beta/api/ws')
    expect(dialed(socket).searchParams.get('relay_ticket')).toBe('relay-ticket')
    expect(dialed(forLocal).pathname).toBe('/api/relay/beta/api/ws')
    expect(ownRequests(requests).every(request => HOST_SURFACE.has(request))).toBe(true)
  })

  it('mints a fresh relay ticket for every redial, as after the host closes the socket with 4401', async () => {
    const requests = dashboardHost()
    const desktop = install()

    await desktop.getGatewayWsUrl()
    await desktop.revalidateConnection!()
    await desktop.getGatewayWsUrl()

    expect(ownRequests(requests).filter(request => request.includes('socket-ticket'))).toEqual([
      'POST /api/webapp/socket-ticket?socket=relay',
      'POST /api/webapp/socket-ticket?socket=relay'
    ])
  })

  it('opens on the first agent when the list names no primary it holds', async () => {
    dashboardHost({ primary: 'gone' })

    expect((await install().getConnection()).connectionId).toBe('alpha')
  })

  it('says so when the account has no agent here', async () => {
    dashboardHost({ agents: [] })

    await expect(install().getConnection()).rejects.toBeInstanceOf(NoHostedAgentError)
  })

  it('lists only the agents in its roster, never a `local` one, and reads no host profiles', async () => {
    const requests = dashboardHost()
    const roster = await install().getAgentRoster!()

    expect(roster.primaryConnectionId).toBe('alpha')
    expect(roster.sources.map(source => source.connectionId)).toEqual(['alpha', 'beta'])
    expect(roster.agents.map(row => `${row.connectionId}/${row.profile}`)).toEqual(['alpha/default', 'beta/default'])
    expect(ownRequests(requests)).not.toContain('GET /api/profiles')
    expect(ownRequests(requests).every(request => HOST_SURFACE.has(request))).toBe(true)
  })

  it("asks the window's agent, not the host, for its version, Portal status and connection test", async () => {
    const requests = dashboardHost()
    const desktop = install()

    expect((await desktop.getVersion()).appVersion).toBe('alpha-1.0')
    expect((await desktop.cloud.status()).signedIn).toBe(true)
    expect(await desktop.connections!.test('beta')).toMatchObject({ ok: true, reachable: true, version: 'beta-1.0' })
    // The desktop-plugins folder is the serving machine's, and this host has none.
    expect(desktop.desktopPluginsRoot).toBeUndefined()
    expect(ownRequests(requests).every(request => HOST_SURFACE.has(request))).toBe(true)
  })

  it('follows the connection boot published for requests that name none', async () => {
    dashboardHost()
    const desktop = install()
    $connection.set(await desktop.getConnectionFor!({ connectionId: 'beta' }))

    expect(await desktop.api({ path: '/api/config' })).toEqual({ from: 'beta', path: '/api/config' })
  })

  it.each([
    ['the host', '/api/gateway-connections'],
    ['a relayed agent', '/api/relay/alpha/api/config']
  ])('takes the whole page to sign-in when %s answers that the session expired', async (_who, path) => {
    dashboardHost({ expired: new Set([path]) })
    const desktop = install()
    const signIn: string[] = []

    window.addEventListener(
      'hermes:browser-reauth-required',
      event => {
        event.preventDefault()
        signIn.push((event as CustomEvent<{ loginUrl: string }>).detail.loginUrl)
      },
      { once: true }
    )

    await expect(desktop.api({ path: '/api/config' })).rejects.toThrow(/session_expired/)
    expect(signIn).toEqual([`${window.location.origin}/login`])
  })
})

describe('a Webapp whose host runs its own agent', () => {
  it.each([undefined, true])('keeps requests that name no connection on the host (flag %s)', async hostAgent => {
    const requests = dashboardHost()
    const desktop = install({ hostAgent })

    await desktop.api({ path: '/api/config' }).catch(() => undefined)

    expect(ownRequests(requests)).toContain('GET /api/config')
    expect((await desktop.getConnection()).connectionId).toBe('local')
  })
})
