import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry, DesktopRegistryConnection } from '@/global'
import { pluginRouteStillRegistered } from '@/sdk/profile-request'
import { $connection } from '@/store/session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { ManagedConnectionsError, NoHostedAgentError } from './browser-saved-connections'
import { _resetAdoptedLaunchProfileForTests, browserLaunchProfile } from './platform'

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
  expired = new Set<string>(),
  profiles = { alpha: ['default'], beta: ['default'] } as Record<string, string[]>,
  runsAs = {} as Record<string, string>
}: {
  agents?: DesktopRegistryConnection[]
  expired?: Set<string>
  primary?: string
  profiles?: Record<string, string[]>
  runsAs?: Record<string, string>
} = {}) {
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
          return json({ profiles: (profiles[id] ?? []).map(name => ({ name })) })
        }

        if (rest === '/api/profiles/active') {
          return json({ current: runsAs[id] ?? 'default' })
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
  _resetAdoptedLaunchProfileForTests()
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
    expect(health).toMatchObject({ current: 'default' })
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

  it('reads the agent list once for roster reads made at the same moment', async () => {
    const requests = dashboardHost()
    const desktop = install()
    await desktop.getConnection()
    requests.length = 0

    await Promise.all([desktop.getAgentRoster!(), desktop.getAgentRoster!()])

    expect(ownRequests(requests)).toEqual(['GET /api/gateway-connections'])
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

  it('sends REST, sockets and profile routes that name no agent to the one the window is on', async () => {
    const requests = dashboardHost({ profiles: { alpha: ['default'], beta: ['default', 'research'] } })
    const desktop = install()
    // Alpha is the primary; the window has moved to Beta.
    $connection.set(await desktop.getConnectionFor!({ connectionId: 'beta' }))

    expect(await desktop.api({ path: '/api/config' })).toEqual({ from: 'beta', path: '/api/config' })
    expect(await desktop.api({ connectionId: 'local', path: '/api/config' })).toEqual({ from: 'beta', path: '/api/config' })
    // A profile named with no agent at all stays on the window socket's agent, as a served Webapp's does.
    expect(dialed(await desktop.getGatewayWsUrl()).pathname).toBe('/api/relay/alpha/api/ws')
    expect(dialed(await desktop.getGatewayWsUrlFor!({ connectionId: 'local' })).pathname).toBe('/api/relay/beta/api/ws')

    const audio = await desktop.getAudioWsUrl!({ connectionId: 'local', path: '/api/audio/transcribe-stream' })
    expect(new URL(audio).pathname).toBe('/api/relay/beta/api/audio/transcribe-stream')
    expect((await desktop.getConnectionFor!({ connectionId: 'local' })).connectionId).toBe('beta')
    expect((await desktop.getProfileRoutes!(['research'])).find(route => route.profile === 'research')).toMatchObject({
      connectionId: 'beta'
    })
    expect((await desktop.getAgentRoster!()).primaryConnectionId).toBe('beta')
    expect(ownRequests(requests).every(request => HOST_SURFACE.has(request))).toBe(true)
  })

  // The store falls back to the window's own socket when the agent in view leaves the account.
  it("falls back to the window socket's agent when the agent in view leaves the account", async () => {
    const agents = [ALPHA, BETA]
    dashboardHost({ agents })
    const desktop = install()
    $connection.set(await desktop.getConnection())
    $connection.set(await desktop.getConnectionFor!({ connectionId: 'beta' }))

    agents.splice(1, 1)
    await desktop.connections!.list()

    // `$connection` still names Beta until the fallback publishes Alpha's descriptor.
    expect(await desktop.getConnection('default')).toMatchObject({ connectionId: 'alpha', profile: 'default' })
    expect(await desktop.api({ path: '/api/config' })).toEqual({ from: 'alpha', path: '/api/config' })
    expect(dialed(await desktop.getGatewayWsUrlFor!({ connectionId: 'local' })).pathname).toBe('/api/relay/alpha/api/ws')
  })

  it('reopens on the next agent in the list when the one the window socket opened on leaves it', async () => {
    const agents = [ALPHA, BETA]
    dashboardHost({ agents })
    const desktop = install()
    $connection.set(await desktop.getConnection())

    agents.splice(0, 1)
    await desktop.connections!.list()

    expect(dialed(await desktop.getGatewayWsUrl()).pathname).toBe('/api/relay/beta/api/ws')
    expect((await desktop.getConnection()).connectionId).toBe('beta')
  })

  it("routes every agent's profiles, the window socket's agent marked primary", async () => {
    dashboardHost({ profiles: { alpha: ['default', 'research'], beta: ['default'] } })
    const routes = await install().getProfileRoutes!(['research', 'deleted', ''])

    expect(routes.map(route => `${route.connectionId}/${route.profile}${route.primary ? ' primary' : ''}`)).toEqual([
      'alpha/default primary',
      'alpha/research primary',
      'beta/default'
    ])
  })

  // A Bot Chat on Alpha that is slow to load retries only while its route is still registered.
  it('still registers a chat route on an agent the window is not on', async () => {
    dashboardHost({ profiles: { alpha: ['default', 'bot'], beta: ['default'] } })
    const desktop = install()
    $connection.set(await desktop.getConnection())
    $connection.set(await desktop.getConnectionFor!({ connectionId: 'beta' }))

    const route = { connectionId: 'alpha', mode: 'remote' as const, profile: 'bot', targetProfile: 'bot' }

    await expect(pluginRouteStillRegistered(route)).resolves.toBe(true)
    await expect(pluginRouteStillRegistered({ ...route, connectionId: 'beta' })).resolves.toBe(false)
  })

  it("adopts the profile the window's agent runs as, as a served page adopts its server's", async () => {
    dashboardHost({ runsAs: { alpha: 'research' } })
    await install().getConnection()

    expect(browserLaunchProfile()).toBe('research')
  })

  // `/api/profiles/active` answers `custom` for an agent run from its own HERMES_HOME: a description
  // of that home, not a profile. The agent's own page names no profile then, and neither does this one.
  it('adopts no profile from an agent that runs from a custom home', async () => {
    dashboardHost({ runsAs: { alpha: 'custom' } })
    await install().getConnection()

    expect(browserLaunchProfile()).toBe('')
  })

  it('refuses to add or remove an agent before sending anything, and says the account manages the list', async () => {
    const requests = dashboardHost()
    const connections = install().connections!

    expect(connections.managedByHost).toBe(true)
    await expect(
      connections.save({ kind: 'remote', label: 'Mine', token: 'secret-token', url: 'https://mine.example' })
    ).rejects.toBeInstanceOf(ManagedConnectionsError)
    await expect(connections.remove('alpha')).rejects.toBeInstanceOf(ManagedConnectionsError)
    expect(requests).toEqual([])
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
