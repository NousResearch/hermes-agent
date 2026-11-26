import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry, DesktopRegistryConnection } from '@/global'
import { pluginRouteStillRegistered } from '@/sdk/profile-request'
import { $desktopBoot } from '@/store/boot'
import { $connection } from '@/store/session'
import { browserOpensWebappWindows } from '@/test/webapp-windows'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { HostedAgentUnavailableError } from './browser-hosted-route'
import { ManagedConnectionsError, NoHostedAgentError } from './browser-saved-connections'
import { signsInToCloudAgents, transferAvailable } from './hermes-cloud-host'
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
  refusedBy = new Set<string>(),
  runsAs = {} as Record<string, string>,
  unreadable = new Set<string>(),
  hung = new Set<string>(),
  serves = {} as Record<string, string[]>
}: {
  agents?: DesktopRegistryConnection[]
  expired?: Set<string>
  /** Agents that refuse the host's exchanged credential: the relay answers their gated routes 409. */
  refusedBy?: Set<string>
  primary?: string
  profiles?: Record<string, string[]>
  runsAs?: Record<string, string>
  /** Agents whose profile list cannot be read (a 500). */
  unreadable?: Set<string>
  /** Agents whose profile list never answers. */
  hung?: Set<string>
  /** The profiles an agent's gateway serves beside its own (`/api/status` `gateway_shared_with`). */
  serves?: Record<string, string[]>
} = {}) {
  const requests: { method: string; path: string; search: string }[] = []

  const profilesOf = (id: string) =>
    hung.has(id)
      ? new Promise<Response>(() => undefined)
      : unreadable.has(id)
        ? json({ detail: 'unavailable' }, 500)
        : json({ profiles: (profiles[id] ?? []).map(name => ({ name })) })

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

        if (refusedBy.has(id)) {
          return json({ detail: 'The agent did not accept this session. Reload to reconnect.', error: 'relay_reconnect' }, 409)
        }

        if (rest === '/api/profiles') {
          return profilesOf(id)
        }

        if (rest === '/api/profiles/active') {
          return json({ current: runsAs[id] ?? 'default' })
        }

        if (rest === '/api/status' && serves[id]) {
          return json({ gateway_shared_with: serves[id] })
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
  $desktopBoot.set({ ...$desktopBoot.get(), error: null })
  window.history.replaceState(null, '', '/')
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

  // The store reuses the window's socket for a profile only when the descriptor says it rides it
  // (`sharedPrimary`); otherwise it dials a second socket to the same agent.
  it("marks a profile the window socket's agent serves as riding that socket, as a served Webapp does", async () => {
    dashboardHost({ profiles: { alpha: ['default', 'research'], beta: ['default'] }, runsAs: { alpha: 'default' } })
    const desktop = install()
    await desktop.getAgentRoster!()

    expect(await desktop.getConnection('research')).toMatchObject({
      connectionId: 'alpha',
      profile: 'research',
      sharedPrimary: true
    })
    // The agent's own profile is the socket itself, not a profile riding it.
    expect((await desktop.getConnection('default')).sharedPrimary).toBe(false)
    expect((await desktop.getConnection()).sharedPrimary).toBe(false)
  })

  it("does not mark a profile the window socket's agent has but does not serve as riding that socket", async () => {
    dashboardHost({
      profiles: { alpha: ['default', 'research', 'parked'], beta: ['default'] },
      runsAs: { alpha: 'default' },
      serves: { alpha: ['research'] }
    })
    const desktop = install()
    await desktop.getAgentRoster!()

    expect((await desktop.getConnection('research')).sharedPrimary).toBe(true)
    expect((await desktop.getConnection('parked')).sharedPrimary).toBe(false)
  })

  it('sends a request naming an agent, made before the list is read, to that agent', async () => {
    dashboardHost()

    await expect(install().api({ connectionId: 'beta', path: '/api/config' })).resolves.toEqual({
      from: 'beta',
      path: '/api/config'
    })
  })

  it('opens on the first agent when the list names no primary it holds', async () => {
    dashboardHost({ primary: 'gone' })

    expect((await install().getConnection()).connectionId).toBe('alpha')
  })

  // A bookmark or a reopened tab naming an agent that left the account: opening the primary in its
  // place would send the user's chats and settings to an agent they did not pick.
  it('fails to open, naming the agent, when the address names one the account no longer lists', async () => {
    dashboardHost()
    window.history.replaceState(null, '', '/?connectionId=gamma&profile=research#/')
    const desktop = install()

    await expect(desktop.getConnection()).rejects.toBeInstanceOf(HostedAgentUnavailableError)
    expect($desktopBoot.get().error).toContain('gamma')
    // Nothing in this page opens another agent in its place.
    await expect(desktop.api({ path: '/api/config' })).rejects.toBeInstanceOf(HostedAgentUnavailableError)
    await expect(desktop.getGatewayWsUrl()).rejects.toBeInstanceOf(HostedAgentUnavailableError)

    // Retry reloads the page, which opens the account's primary: the address names it, not gamma.
    const address = new URL(window.location.href)
    expect([address.searchParams.get('connectionId'), address.searchParams.get('profile')]).toEqual(['alpha', null])
    Reflect.deleteProperty(win, 'hermesDesktop')
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

  // The window's socket cannot follow its agent out of the account, and moving requests to another
  // agent while that socket stays on the removed one splits the window across two agents.
  it('fails the window, naming the agent, when the one its socket opened on leaves the account', async () => {
    const agents = [ALPHA, BETA]
    const requests = dashboardHost({ agents })
    const desktop = install()
    $connection.set(await desktop.getConnection())

    agents.splice(0, 1)
    await desktop.connections!.list()
    requests.length = 0

    expect($desktopBoot.get().error).toContain('alpha')
    await expect(desktop.getGatewayWsUrl()).rejects.toBeInstanceOf(HostedAgentUnavailableError)
    await expect(desktop.getConnection()).rejects.toBeInstanceOf(HostedAgentUnavailableError)
    await expect(desktop.api({ path: '/api/config' })).rejects.toBeInstanceOf(HostedAgentUnavailableError)
    expect(requests.filter(request => request.path.startsWith('/api/relay/beta/'))).toEqual([])
  })

  // The chat in the address is the removed agent's: Retry opens the primary on a new chat, not on a
  // resume of that chat id there, and the address names the primary so startup's last-used
  // preference cannot send the window elsewhere.
  it.each([
    ['a popped-out chat', '/?win=secondary&profile=research&connectionId=beta&watch=1#/s1', 'beta', 'alpha'],
    ['a window on the primary', '/?profile=default#/s1', 'alpha', 'beta']
  ])("drops the removed agent's chat from the address of %s", async (_window, href, removed, primary) => {
    const agents = [ALPHA, BETA]
    dashboardHost({ agents, profiles: { alpha: ['default'], beta: ['default', 'research'] } })
    window.history.replaceState(null, '', href)
    const desktop = install()
    $connection.set(await desktop.getConnection())
    expect($connection.get()?.connectionId).toBe(removed)

    agents.splice(agents.findIndex(row => row.id === removed), 1)
    await desktop.connections!.list()

    expect($desktopBoot.get().error).toContain(removed)
    const address = new URL(window.location.href)
    expect(['win', 'profile', 'connectionId', 'watch'].map(name => address.searchParams.get(name))).toEqual([
      null,
      null,
      primary,
      null
    ])
    expect(address.hash).not.toContain('s1')
    window.history.replaceState(null, '', '/')
  })

  // A connection pick moves the window without writing its agent into the address.
  it.each([
    ['names no agent', 'profile=default'],
    ['still names the agent that left', 'connectionId=alpha&profile=default']
  ])("keeps the chat route of a window that moved to another agent when its address %s and the one its socket opened on leaves", async (_case, launch) => {
    const agents = [ALPHA, BETA]
    dashboardHost({ agents })
    window.history.replaceState(null, '', `/?${launch}`)
    const desktop = install()
    $connection.set(await desktop.getConnection())
    $connection.set(await desktop.getConnectionFor!({ connectionId: 'beta', profile: 'research' }))
    window.history.replaceState(null, '', `/?${launch}#/beta-chat`)

    agents.splice(0, 1)
    await desktop.connections!.list()

    const address = new URL(window.location.href)
    expect([address.searchParams.get('connectionId'), address.searchParams.get('profile'), address.hash]).toEqual([
      'beta',
      'research',
      '#/beta-chat'
    ])
  })

  it('keeps the agent the window moved to in its address when the one its socket opened on leaves', async () => {
    const agents = [ALPHA, BETA]
    dashboardHost({ agents })
    const desktop = install()
    $connection.set(await desktop.getConnection())
    $connection.set(await desktop.getConnectionFor!({ connectionId: 'beta' }))
    await desktop.profile!.remember(null)

    agents.splice(0, 1)
    await desktop.connections!.list()

    // Retry reloads the page onto the agent the window was on.
    expect(new URL(window.location.href).searchParams.get('connectionId')).toBe('beta')
  })

  // The account's primary can change in another tab while this one is open: a chat popped out of
  // this window must open on its own agent, not on whichever agent the list names primary by then.
  it("pops a chat out into a window that opens on the chat's agent", async () => {
    dashboardHost({ primary: 'alpha', profiles: { alpha: ['default'], beta: ['default', 'research'] } })
    const desktop = install()
    $connection.set(await desktop.getConnection())
    $connection.set(await desktop.getConnectionFor!({ connectionId: 'beta', profile: 'research' }))
    const { open } = browserOpensWebappWindows()

    await expect(desktop.openSessionWindow('s1', { connectionId: 'beta', profile: 'research' })).resolves.toEqual({ ok: true })
    await expect(desktop.openSessionWindow('s2', { profile: 'research' })).resolves.toEqual({ ok: true })

    const opened = open.mock.calls.map(([url]) => new URL(String(url)))
    expect(opened.map(url => url.searchParams.get('connectionId'))).toEqual(['beta', 'beta'])

    // The new window boots on the agent its address names.
    Reflect.deleteProperty(win, 'hermesDesktop')
    $connection.set(null)
    window.history.replaceState(null, '', `${opened[0].pathname}${opened[0].search}${opened[0].hash}`)
    const popped = install()

    expect((await popped.getConnection()).connectionId).toBe('beta')
    expect(dialed(await popped.getGatewayWsUrl('research')).pathname).toBe('/api/relay/beta/api/ws')
    window.history.replaceState(null, '', '/')
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

  // A switch writes the profile into the address; on this host it writes the agent beside it, or a
  // reload of a popped-out window would open the profile on the agent the window was opened for.
  it('keeps the agent the window switched to in its address, so a reload opens on it', async () => {
    dashboardHost({ profiles: { alpha: ['default', 'research'], beta: ['default'] } })
    window.history.replaceState(null, '', '/?win=secondary&connectionId=beta')
    const desktop = install()
    $connection.set(await desktop.getConnection())
    expect($connection.get()?.connectionId).toBe('beta')

    $connection.set(await desktop.getConnectionFor!({ connectionId: 'alpha', profile: 'research' }))
    await desktop.profile!.remember('research')

    const address = new URL(window.location.href)
    expect([address.searchParams.get('connectionId'), address.searchParams.get('profile')]).toEqual(['alpha', 'research'])

    Reflect.deleteProperty(win, 'hermesDesktop')
    $connection.set(null)
    const reloaded = install()

    expect((await reloaded.getConnection()).connectionId).toBe('alpha')
    window.history.replaceState(null, '', '/')
  })

  it('opens a new window on the agent this one is on, or the one the route names', async () => {
    dashboardHost({ profiles: { alpha: ['default'], beta: ['default', 'research'] } })
    const desktop = install()
    $connection.set(await desktop.getConnection())
    $connection.set(await desktop.getConnectionFor!({ connectionId: 'beta', profile: 'research' }))
    const { open } = browserOpensWebappWindows()

    await desktop.openWindow!()
    await desktop.openWindow!({ connectionId: 'alpha', profile: 'default' })

    const opened = open.mock.calls.map(([url]) => new URL(String(url)))
    expect(opened.map(url => `${url.searchParams.get('connectionId')}/${url.searchParams.get('profile')}`)).toEqual([
      'beta/research',
      'alpha/default'
    ])
  })

  // Boot drops the address's profile when the window's own routes lack it; an unreadable list must
  // not read as one that lacks it.
  it("fails the window's own route read when the window socket's agent cannot list its profiles", async () => {
    dashboardHost({ profiles: { alpha: ['default', 'research'], beta: ['default'] }, unreadable: new Set(['alpha']) })

    await expect(install().getProfileRoutes!(['research'], { own: true })).rejects.toThrow()
  })

  // A Bot Chat on Beta that is slow to load retries while its route is still registered: Alpha's
  // failure says nothing about Beta.
  it("still registers a chat route on a healthy agent when the window socket's agent cannot list its profiles", async () => {
    dashboardHost({ profiles: { alpha: ['default'], beta: ['default', 'bot'] }, unreadable: new Set(['alpha']) })
    install()

    const route = { connectionId: 'beta', mode: 'remote' as const, profile: 'bot', targetProfile: 'bot' }

    await expect(pluginRouteStillRegistered(route)).resolves.toBe(true)
  })

  // Boot checks the address's profile against the window's own routes only; another agent that
  // never answers must not hold that check up for the roster timeout.
  it("answers the window's own routes without waiting on another agent", async () => {
    dashboardHost({ hung: new Set(['beta']), profiles: { alpha: ['default', 'research'], beta: ['default'] } })
    const desktop = install()
    $connection.set(await desktop.getConnection())
    const late = new Promise(resolve => setTimeout(() => resolve('timed out'), 500))

    const routes = await Promise.race([desktop.getProfileRoutes!(['research'], { own: true }), late])

    expect(routes).toEqual([
      { connectionId: 'alpha', mode: 'remote', primary: true, profile: 'default', targetProfile: 'default' },
      { connectionId: 'alpha', mode: 'remote', primary: true, profile: 'research', targetProfile: 'research' }
    ])
  })

  it("fails the window's own route read when the account list cannot be read after boot", async () => {
    dashboardHost({ profiles: { alpha: ['default', 'research'], beta: ['default'] } })
    const desktop = install()
    $connection.set(await desktop.getConnection())
    const host = vi.mocked(fetch).getMockImplementation()!
    vi.mocked(fetch).mockImplementation(async (url, init) =>
      new URL(String(url), window.location.origin).pathname === '/api/gateway-connections'
        ? new Response('{}', { status: 503 })
        : host(url, init)
    )

    await expect(desktop.getProfileRoutes!(['research'], { own: true })).rejects.toThrow()
  })

  it("still routes the window socket's agent when another agent cannot list its profiles", async () => {
    dashboardHost({ profiles: { alpha: ['default', 'research'], beta: ['default'] }, unreadable: new Set(['beta']) })
    const routes = await install().getProfileRoutes!(['research'])

    expect(routes.map(route => `${route.connectionId}/${route.profile}`)).toEqual(['alpha/default', 'alpha/research'])
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

  // The socket opens on the profile the agent runs as. A read that failed is not an agent that names
  // no profile: taken as one, boot would record `default` while the socket opens on `research`.
  it.each([
    [1, 'research'],
    [2, undefined]
  ])("reads the profile the window's agent runs as again, and fails the open after %i failed reads", async (fails, adopted) => {
    dashboardHost({ runsAs: { alpha: 'research' } })
    const host = vi.mocked(fetch).getMockImplementation()!
    let failed = 0

    vi.mocked(fetch).mockImplementation(async (url, init) => {
      if (new URL(String(url), window.location.origin).pathname === '/api/relay/alpha/api/profiles/active' && failed < fails) {
        failed += 1

        return json({ detail: 'timed out' }, 504)
      }

      return host(url, init)
    })

    const open = install().getConnection()

    if (adopted) {
      await expect(open).resolves.toMatchObject({ connectionId: 'alpha' })
    } else {
      await expect(open).rejects.toThrow()
    }

    expect(browserLaunchProfile()).toBe(adopted ?? '')
  })

  // A failed read is not kept: once the agent answers again, the next open adopts its profile.
  it("opens and adopts the agent's profile after an open that failed its reads", async () => {
    dashboardHost({ runsAs: { alpha: 'research' } })
    const host = vi.mocked(fetch).getMockImplementation()!
    let down = true

    vi.mocked(fetch).mockImplementation(async (url, init) =>
      down && new URL(String(url), window.location.origin).pathname === '/api/relay/alpha/api/profiles/active'
        ? json({ detail: 'timed out' }, 504)
        : host(url, init)
    )
    const desktop = install()
    await expect(desktop.getConnection()).rejects.toThrow()

    down = false

    await expect(desktop.getConnection()).resolves.toMatchObject({ connectionId: 'alpha' })
    expect(browserLaunchProfile()).toBe('research')
  })

  // `/api/profiles/active` answers `custom` for an agent run from its own HERMES_HOME: a description
  // of that home, not a profile. The agent's own page names no profile then, and neither does this one.
  it('adopts no profile from an agent that runs from a custom home', async () => {
    dashboardHost({ runsAs: { alpha: 'custom' } })
    await install().getConnection()

    expect(browserLaunchProfile()).toBe('')
  })

  // `/api/health` is public on an agent, so it answers whether or not the agent takes the session.
  it("fails a connection test when the agent refuses the host's credential", async () => {
    dashboardHost({ refusedBy: new Set(['beta']) })
    const connections = install().connections!
    await connections.list()

    await expect(connections.test('beta')).resolves.toMatchObject({ ok: false })
    await expect(connections.test('alpha')).resolves.toMatchObject({ ok: true, version: 'alpha-1.0' })
  })

  it('signs in to no cloud agent, and Settings offers no Transfer that needs one', async () => {
    const requests = dashboardHost()
    const desktop = install()

    expect(await desktop.cloud.agentSignIn('https://gamma.agents.example')).toEqual({
      baseUrl: 'https://gamma.agents.example',
      connected: false
    })
    expect(requests).toEqual([])
    expect(signsInToCloudAgents()).toBe(false)
    expect(transferAvailable()).toBe(false)
  })

  it('offers Transfer on a Webapp whose host runs its own agent', () => {
    dashboardHost()
    install({ hostAgent: undefined })

    expect(signsInToCloudAgents()).toBe(true)
    expect(transferAvailable()).toBe(true)
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

  it("tells a switch that names no profile the profile the picked agent runs as, through that agent's relay", async () => {
    const requests = dashboardHost({ runsAs: { beta: 'research' } })
    const connections = install().connections!

    await connections.list()
    expect(await connections.launchProfile?.('beta')).toBe('research')
    expect(requests.map(request => request.path)).toContain('/api/relay/beta/api/profiles/active')
  })

  it('leaves the landing profile to the store on a Webapp whose host runs its own agent', () => {
    dashboardHost()

    expect(install({ hostAgent: undefined }).connections?.launchProfile).toBeUndefined()
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
