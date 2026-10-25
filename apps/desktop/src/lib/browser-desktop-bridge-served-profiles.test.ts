import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { resolveDesktopGatewayWsUrl } from './gateway-ws-url'

type ServedWindow = Window & { __HERMES_DASHBOARD_PROFILE__?: string; hermesDesktop?: Window['hermesDesktop'] }

const win = window as unknown as ServedWindow

// One gateway's answers: `/api/profiles` lists every installed profile, `/api/status` names the
// ones this process serves. Tests mutate it to change what the gateway reports.
interface Gateway {
  profiles: string[]
  status: (init?: RequestInit) => Promise<Response> | Response
}

function stubGateway(gateway: Gateway) {
  const fetchMock = vi.fn(async (url: URL, init?: RequestInit) => {
    if (url.pathname.endsWith('/api/status')) {
      return gateway.status(init)
    }

    const body = url.pathname.endsWith('/api/webapp/socket-ticket')
      ? { ticket: 'one-use' }
      : { profiles: gateway.profiles.map(name => ({ name })) }

    return new Response(JSON.stringify(body), { status: 200 })
  })

  vi.stubGlobal('fetch', fetchMock)

  return fetchMock
}

const reports = (shared: null | string[]) => () =>
  new Response(JSON.stringify({ gateway_running: true, gateway_shared_with: shared }), { status: 200 })

const callsTo = (fetchMock: ReturnType<typeof stubGateway>, path: string) =>
  fetchMock.mock.calls.filter(([url]) => url.pathname.endsWith(path)).length

function installServedBridge(launchProfile = 'default') {
  win.__HERMES_DASHBOARD_PROFILE__ = launchProfile
  seedWebappSession('served-token')
  expect(installBrowserDesktopBridge()).toBe(true)

  return win.hermesDesktop!
}

afterEach(() => {
  delete win.__HERMES_DASHBOARD_PROFILE__
  clearWebappSession()
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  $connection.set(null)
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('profiles the serving gateway does not serve', () => {
  it('lists them on the roster as unreachable and routes none of them over the tab socket', async () => {
    stubGateway({ profiles: ['default', 'research', 'solo'], status: reports(['default', 'research']) })
    const desktop = installServedBridge()

    await expect(desktop.getProfileRoutes?.(['default', 'research', 'solo'])).resolves.toEqual([
      { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' },
      { connectionId: 'local', mode: 'local', profile: 'research', targetProfile: 'research' }
    ])

    const roster = await desktop.getAgentRoster?.()
    expect(roster?.sources).toEqual([expect.objectContaining({ connectionId: 'local', kind: 'local', reachable: true })])
    expect(roster?.agents.find(agent => agent.profile === 'research')).toEqual(expect.not.objectContaining({ reachable: false }))
    expect(roster?.agents.find(agent => agent.profile === 'solo')).toMatchObject({
      connectionId: 'local',
      profile: 'solo',
      reachable: false
    })

    await expect(desktop.getConnection('research')).resolves.toMatchObject({ profile: 'research', sharedPrimary: true })
    await expect(desktop.getConnection('solo')).resolves.toMatchObject({ profile: 'solo', sharedPrimary: false })
    await expect(desktop.getConnectionFor?.({ connectionId: 'local', profile: 'research' })).resolves.toMatchObject({
      sharedRemote: true
    })
    await expect(desktop.getConnectionFor?.({ connectionId: 'local', profile: 'solo' })).resolves.toMatchObject({
      sharedPrimary: false,
      sharedRemote: false
    })
  })

  it('reads an empty served list as a gateway that serves only the profile it runs as', async () => {
    stubGateway({ profiles: ['default', 'research'], status: reports([]) })
    const desktop = installServedBridge('research')

    await expect(desktop.getProfileRoutes?.(['default', 'research'])).resolves.toEqual([
      { connectionId: 'local', mode: 'local', profile: 'research', targetProfile: 'research' }
    ])

    const roster = await desktop.getAgentRoster?.()
    expect(roster?.agents.find(agent => agent.profile === 'research')).toEqual(expect.not.objectContaining({ reachable: false }))
    expect(roster?.agents.find(agent => agent.profile === 'default')?.reachable).toBe(false)
    await expect(desktop.getConnection('default')).resolves.toMatchObject({ sharedPrimary: false })
    await expect(desktop.getConnectionFor?.({ connectionId: 'local', profile: 'default' })).resolves.toMatchObject({
      sharedRemote: false
    })
    await expect(desktop.getConnectionFor?.({ connectionId: 'local', profile: 'research' })).resolves.toMatchObject({
      sharedRemote: true
    })
  })

  it('keeps every installed profile reachable when the gateway does not report its served set', async () => {
    stubGateway({ profiles: ['default', 'research'], status: reports(null) })
    const desktop = installServedBridge()

    await expect(desktop.getProfileRoutes?.(['default', 'research'])).resolves.toHaveLength(2)
    const roster = await desktop.getAgentRoster?.()
    expect(roster?.agents.every(agent => agent.reachable !== false)).toBe(true)
    await expect(desktop.getConnection('research')).resolves.toMatchObject({ sharedPrimary: true })
    await expect(desktop.getConnectionFor?.({ connectionId: 'local', profile: 'research' })).resolves.toMatchObject({
      sharedRemote: true
    })
  })

  it('refuses a dial to one with the reason, before minting a ticket the gateway would reject', async () => {
    const fetchMock = stubGateway({ profiles: ['default', 'research', 'solo'], status: reports(['default', 'research']) })
    const desktop = installServedBridge()
    await desktop.getProfileRoutes?.(['solo'])

    const refusal = 'This gateway does not serve profile "solo", so this tab cannot open it.'
    await expect(resolveDesktopGatewayWsUrl(desktop, await desktop.getConnection('solo'))).rejects.toThrow(refusal)
    await expect(
      resolveDesktopGatewayWsUrl(desktop, await desktop.getConnectionFor!({ connectionId: 'local', profile: 'solo' }))
    ).rejects.toThrow(refusal)
    expect(callsTo(fetchMock, '/api/webapp/socket-ticket')).toBe(0)

    // The tab's own socket and a served profile still mint their tickets.
    await expect(resolveDesktopGatewayWsUrl(desktop, await desktop.getConnection())).resolves.toContain('ticket=one-use')
    await expect(desktop.getGatewayWsUrl('research')).resolves.toMatchObject({ ok: true })
    expect(callsTo(fetchMock, '/api/webapp/socket-ticket')).toBe(2)
  })
})

describe('the served set read', () => {
  it('is never fetched to install the bridge or to route a request', async () => {
    const fetchMock = stubGateway({ profiles: ['default', 'owner'], status: reports(['default']) })
    const desktop = installServedBridge()

    await desktop.getConnection()
    await desktop.getConnection('default')
    await expect(desktop.getConnection('owner')).resolves.toMatchObject({ sharedPrimary: true })
    await desktop.getConnectionFor?.({ connectionId: 'local', profile: 'owner' })
    expect(fetchMock).not.toHaveBeenCalled()

    // Once an inventory read learns the set, routing uses it, still without a request of its own.
    await desktop.getProfileRoutes?.(['owner'])
    const reads = fetchMock.mock.calls.length
    await expect(desktop.getConnection('owner')).resolves.toMatchObject({ sharedPrimary: false })
    await desktop.getConnectionFor?.({ connectionId: 'local', profile: 'owner' })
    expect(fetchMock).toHaveBeenCalledTimes(reads)
  })

  it('is shared by inventory reads until the installed list changes, the gateway reconnects or 30 s pass', async () => {
    const gateway: Gateway = { profiles: ['default', 'research', 'solo'], status: reports(['default', 'research']) }
    const fetchMock = stubGateway(gateway)
    const now = vi.spyOn(Date, 'now').mockReturnValue(1_000_000)
    const desktop = installServedBridge()
    const statusReads = () => callsTo(fetchMock, '/api/status')

    await desktop.getProfileRoutes?.(['research'])
    await desktop.getAgentRoster?.()
    await desktop.getProfileRoutes?.(['solo'])
    expect(statusReads()).toBe(1)

    // A new profile: the multiplexer adopts it before its create returns, so read the set again.
    gateway.profiles = [...gateway.profiles, 'fresh']
    gateway.status = reports(['default', 'research', 'fresh'])
    await expect(desktop.getProfileRoutes?.(['fresh'])).resolves.toHaveLength(1)
    expect(statusReads()).toBe(2)

    // A park or unpark leaves the list alone and shows up on the next read after 30 s.
    gateway.status = reports(['default', 'research', 'fresh', 'solo'])
    now.mockReturnValue(1_029_000)
    await expect(desktop.getProfileRoutes?.(['solo'])).resolves.toEqual([])
    now.mockReturnValue(1_031_000)
    await expect(desktop.getProfileRoutes?.(['solo'])).resolves.toHaveLength(1)
    expect(statusReads()).toBe(3)

    // A gateway that restarted may serve another set.
    gateway.status = reports(['default'])
    await desktop.revalidateConnection()
    await expect(desktop.getProfileRoutes?.(['research'])).resolves.toEqual([])
    expect(statusReads()).toBe(4)
  })

  it('keeps the last answer when a later read fails', async () => {
    const gateway: Gateway = { profiles: ['default', 'research', 'solo'], status: reports(['default', 'research']) }
    stubGateway(gateway)
    const desktop = installServedBridge()

    await desktop.getProfileRoutes?.(['solo'])
    gateway.status = () => new Response('{"detail":"gateway unavailable"}', { status: 503 })
    await desktop.revalidateConnection()

    await expect(desktop.getProfileRoutes?.(['research', 'solo'])).resolves.toEqual([
      { connectionId: 'local', mode: 'local', profile: 'research', targetProfile: 'research' }
    ])
    await expect(desktop.getConnection('solo')).resolves.toMatchObject({ sharedPrimary: false })
  })

  it('holds an inventory read back at most 5 s, then answers as if every profile is served', async () => {
    vi.useFakeTimers()
    stubGateway({
      profiles: ['default', 'solo'],
      // A gateway too busy to answer: only the request's own deadline ends the read.
      status: init =>
        new Promise<Response>((_resolve, reject) =>
          init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
        )
    })

    const desktop = installServedBridge()
    let routes: unknown = null
    void desktop.getProfileRoutes?.(['default', 'solo']).then(value => (routes = value))

    // Routing does not wait for the read in flight.
    await expect(desktop.getConnection('solo')).resolves.toMatchObject({ sharedPrimary: true })

    await vi.advanceTimersByTimeAsync(4_900)
    expect(routes).toBeNull()
    await vi.advanceTimersByTimeAsync(100)
    expect(routes).toEqual([
      { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' },
      { connectionId: 'local', mode: 'local', profile: 'solo', targetProfile: 'solo' }
    ])
  })
})
