import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry, DesktopRegistryConnection } from '@/global'
import { $defaultProfileRoute, setDefaultProfile } from '@/store/default-profile'
import { $connection, workspaceCwdForNewSession } from '@/store/session'
import { dropTilesForProfile } from '@/store/session-states'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { dropBrowserProfilePrefs, migrateBrowserProfilePrefs } from './browser-profile-prefs'
import { RELAYED_ROSTER_TIMEOUT_MS } from './browser-saved-connections'
import { UnservedProfileError } from './browser-served-profiles'

type RelayWindow = Window & {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_BASE_PATH__?: string
  __HERMES_UI_SURFACE__?: string
  hermesDesktop?: Window['hermesDesktop']
}

const win = window as unknown as RelayWindow

const LOCAL: DesktopRegistryConnection = {
  id: 'local',
  kind: 'local',
  label: 'Hub',
  tokenPreview: null,
  tokenSet: false
}

const MINI: DesktopRegistryConnection = {
  authMode: 'token',
  headerNames: [],
  id: 'mini',
  kind: 'remote',
  label: 'Mini',
  tokenPreview: '...abcdef',
  tokenSet: true,
  url: 'https://mini.example.ts.net'
}

const registryWith = (rows: DesktopRegistryConnection[]): DesktopConnectionsRegistry => ({
  connections: [LOCAL, ...rows],
  lastUsed: 'local',
  launchMode: 'primary',
  primary: 'local',
  secureTokenStorage: true,
  version: 2
})

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' }, status })

/**
 * The serving gateway at /hermes, whose machine saved Mini; answers by path and records every
 * request. A stalled Mini takes the request and never answers it. `miniServes` is the served set
 * Mini reports (its multiplexer's `gateway_shared_with`); left out, Mini reports none, as an older
 * one does. `miniProfiles` is what Mini has installed. A test changes either array to change Mini,
 * and changes `mini` to point the saved entry elsewhere (`row`) or have it run as another profile.
 */
function servingGateway({
  mini = { own: 'default', row: MINI },
  miniProfiles = ['default', 'research'],
  miniServes,
  miniStalled = false
}: {
  mini?: { own: string; row: DesktopRegistryConnection }
  miniProfiles?: string[]
  miniServes?: string[]
  miniStalled?: boolean
} = {}) {
  const requests: { body?: unknown; method: string; path: string; search: string }[] = []
  const saved = { rows: [mini.row] }

  const fetchMock = vi.fn(async (url: URL, init?: RequestInit) => {
    const path = url.pathname
    const method = init?.method || 'GET'
    requests.push({ body: init?.body ? JSON.parse(String(init.body)) : undefined, method, path, search: url.search })
    saved.rows = saved.rows.map(row => (row.id === 'mini' ? mini.row : row))

    if (path === '/hermes/api/gateway-connections' && method === 'GET') {
      return json(registryWith(saved.rows))
    }

    if (path.startsWith('/hermes/api/gateway-connections') && method === 'POST') {
      return json({ ok: true, registry: registryWith(saved.rows) })
    }

    if (path === '/hermes/api/gateway-connections/mini' && method === 'DELETE') {
      saved.rows = []

      return json({ ok: true, registry: registryWith(saved.rows) })
    }

    if (path === '/hermes/api/relay/mini/api/profiles') {
      if (miniStalled) {
        return new Promise<Response>((_resolve, reject) =>
          init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
        )
      }

      return json({ profiles: miniProfiles.map(name => ({ name })) })
    }

    if (path === '/hermes/api/relay/mini/api/profiles/sessions') {
      return json({ sessions: [{ id: 'm1', profile: 'research', title: 'On Mini' }], total: 1 })
    }

    if (path === '/hermes/api/relay/mini/api/status') {
      return json(miniServes ? { gateway_shared_with: miniServes } : {})
    }

    if (path === '/hermes/api/relay/mini/api/profiles/active') {
      return json({ active: mini.own, current: mini.own })
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

  return Object.assign(requests, { saved })
}

function install({ surface = 'webapp' } = {}) {
  win.__HERMES_AUTH_REQUIRED__ = true
  win.__HERMES_BASE_PATH__ = '/hermes'
  win.__HERMES_UI_SURFACE__ = surface

  expect(installBrowserDesktopBridge()).toBe(true)

  return win.hermesDesktop!
}

afterEach(() => {
  delete win.__HERMES_AUTH_REQUIRED__
  delete win.__HERMES_BASE_PATH__
  delete win.__HERMES_UI_SURFACE__
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  window.localStorage.clear()
  $connection.set(null)
  $defaultProfileRoute.set(null)
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('a Webapp on a machine with saved gateway connections', () => {
  it("lists the machine's saved connections with native Desktop's ids", async () => {
    servingGateway()
    const registry = await install().connections!.list()

    expect(registry.connections.map(connection => [connection.id, connection.kind])).toEqual([
      ['local', 'local'],
      ['mini', 'remote']
    ])
    expect(registry.primary).toBe('local')
  })

  it('saves, removes, sets the primary and the launch mode on the one list, and reports each change', async () => {
    const requests = servingGateway()
    const desktop = install()
    await desktop.connections!.list()
    const events: string[] = []
    desktop.connections!.onChanged!(event => events.push(`${event.connectionId}:${event.reason}`))

    await desktop.connections!.setLaunchMode!('last-used')
    await desktop.connections!.setPrimary('mini')
    await desktop.connections!.save({ kind: 'remote', label: 'Mini', token: 'tok', url: MINI.url })
    await desktop.connections!.remove('mini')

    expect(requests.filter(r => r.method !== 'GET').map(r => [r.method, r.path, r.body])).toEqual([
      ['POST', '/hermes/api/gateway-connections/launch-mode', { mode: 'last-used' }],
      ['POST', '/hermes/api/gateway-connections/mini/primary', undefined],
      ['POST', '/hermes/api/gateway-connections', { kind: 'remote', label: 'Mini', token: 'tok', url: MINI.url }],
      ['DELETE', '/hermes/api/gateway-connections/mini', undefined]
    ])
    expect(events).toContain('mini:removed')
  })

  it('reports a connection native Desktop re-pointed as updated, so its sockets redial', async () => {
    const mini = { own: 'default', row: MINI }
    servingGateway({ mini })
    const desktop = install()
    await desktop.connections!.list()
    const events: string[] = []
    desktop.connections!.onChanged!(event => events.push(`${event.connectionId}:${event.reason}`))

    mini.row = { ...MINI, label: 'Mini PC' }
    await desktop.connections!.list()
    mini.row = { ...MINI, label: 'Mini PC', url: 'https://mini-2.example.ts.net' }
    await desktop.connections!.list()

    expect(events).toEqual(['mini:saved', 'mini:updated'])
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

  it("dials a saved gateway's socket with only the serving gateway's relay ticket", async () => {
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
    // The serving gateway mints Mini's own credential for the dial; none rides the browser's URL.
    expect(url.searchParams.has('ticket')).toBe(false)
    expect(url.searchParams.has('native_dial')).toBe(false)
    expect(url.searchParams.get('relay_ticket')).toBe('hub-relay')
    expect(url.searchParams.get('profile')).toBe('research')

    // The descriptor carries no credential; each dial mints its own pair.
    const connection = await desktop.getConnectionFor!({ connectionId: 'mini', profile: 'research' })
    expect(connection.connectionId).toBe('mini')
    expect(new URL(connection.wsUrl).searchParams.has('ticket')).toBe(false)
  })

  it('follows the window onto a saved gateway for its host terminal and files', async () => {
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
    expect(dialed[0]?.searchParams.has('ticket')).toBe(false)
    expect(dialed[0]?.searchParams.get('relay_ticket')).toBe('hub-relay')

    const stream = new URL(await desktop.getGatewayFileStreamUrl!({ connectionId: 'mini', path: '/tmp/a.png' }))
    expect(stream.pathname.startsWith('/hermes/api/relay/mini/')).toBe(true)
  })

  it('closes a terminal on the gateway it started on after the window moves to another', async () => {
    servingGateway()
    const desktop = install()
    await desktop.connections!.list()
    const onMini = { ...(await desktop.getConnectionFor!({ connectionId: 'mini' })) }
    $connection.set(onMini)
    const dialed: URL[] = []
    const META = '\u0000HERMES_TERMINAL_META:'

    vi.stubGlobal(
      'WebSocket',
      class {
        static OPEN = 1
        readyState = 1
        binaryType = 'blob'
        onclose: ((event: CloseEvent) => void) | null = null
        onerror: (() => void) | null = null
        onmessage: ((event: MessageEvent) => void) | null = null
        constructor(url: string | URL) {
          const target = new URL(url)
          dialed.push(target)
          // A shell starts as `t1`; a close dial is acknowledged, then closes normally.
          const closing = target.searchParams.get('action') === 'close'

          queueMicrotask(() => {
            const meta = closing ? { closed: true, terminalId: 't1' } : { cwd: '/', shell: 'bash', terminalId: 't1' }
            this.onmessage?.(new MessageEvent('message', { data: META + JSON.stringify(meta) }))

            if (closing) {
              this.onclose?.(new CloseEvent('close', { code: 1000 }))
            }
          })
        }
        close() {}
        send() {}
      }
    )

    const started = await desktop.terminal.start({ restoreKey: 'tab-1' })
    // A new chat on this gateway's own sidebar "+" moves the window; the shell stays on Mini.
    $connection.set({ ...onMini, connectionId: 'local' })
    await desktop.terminal.dispose(started.id)

    expect(dialed.map(url => [url.pathname, url.searchParams.get('action')])).toEqual([
      ['/hermes/api/relay/mini/api/host-terminal', null],
      ['/hermes/api/relay/mini/api/host-terminal', 'close']
    ])
    expect(dialed[1]?.searchParams.get('attach')).toBe('t1')
  })

  it("loads an HTML preview's own scripts from this gateway while the window is on a relayed one", async () => {
    servingGateway()
    const desktop = install()
    await desktop.connections!.list()
    $connection.set({ ...(await desktop.getConnectionFor!({ connectionId: 'mini' })) })
    const blobs: Blob[] = []

    vi.spyOn(URL, 'createObjectURL').mockImplementation(blob => {
      blobs.push(blob as Blob)

      return 'blob:https://hub.example/preview'
    })

    // The wrapper is a document on this origin, so its outer script runs as this gateway's page.
    await desktop.saveImageBuffer(new TextEncoder().encode('<p>hi</p>'), '.html')
    const wrapper = new DOMParser().parseFromString(await blobs[0]!.text(), 'text/html')

    expect(new URL(wrapper.querySelector('script')!.src).pathname).toBe('/hermes/assets/webapp-preview.js')
    expect(new URL(wrapper.querySelector('iframe')!.src).pathname).toBe('/hermes/assets/webapp-preview.html')
  })

  it("keeps the default folder a saved gateway's profile chose apart from this gateway's", async () => {
    servingGateway()
    const desktop = install()
    await desktop.connections!.list()
    const onMini = { ...(await desktop.getConnectionFor!({ connectionId: 'mini', profile: 'research' })) }
    const onHub = { ...onMini, connectionId: 'local' }

    $connection.set(onHub)
    await desktop.settings.setDefaultProjectDir('/srv/hub')
    $connection.set(onMini)
    await desktop.settings.setDefaultProjectDir('/srv/mini')
    expect(workspaceCwdForNewSession('research')).toBe('/srv/mini')

    $connection.set(onHub)
    expect((await desktop.settings.getDefaultProjectDir()).dir).toBe('/srv/hub')
    expect(workspaceCwdForNewSession('research')).toBe('/srv/hub')

    // Renaming the profile on Mini takes Mini's folder with it and leaves this gateway's alone.
    $connection.set(onMini)
    migrateBrowserProfilePrefs('research', 'notes')
    expect(workspaceCwdForNewSession('notes')).toBe('/srv/mini')
    $connection.set(onHub)
    expect(workspaceCwdForNewSession('research')).toBe('/srv/hub')
    expect(workspaceCwdForNewSession('notes')).toBe('')
  })

  it("keeps this gateway's saved profile prefs when a same-named profile changes on a relayed one", async () => {
    servingGateway()
    const desktop = install()
    await desktop.connections!.list()
    const onMini = { ...(await desktop.getConnectionFor!({ connectionId: 'mini', profile: 'research' })) }
    const onHub = { ...onMini, connectionId: 'local' }
    const hubDefault = { connectionId: 'local', profile: 'research' }

    $connection.set(onHub)
    await desktop.settings.setDefaultProjectDir('/srv/hub')
    await setDefaultProfile(hubDefault)
    $connection.set(onMini)
    await desktop.settings.setDefaultProjectDir('/srv/mini')

    // Creating `research` on Mini clears what an earlier Mini `research` left there, nothing here.
    dropBrowserProfilePrefs('research')
    expect(workspaceCwdForNewSession('research')).toBe('')
    // Deleting it from a window on this gateway names the route it was deleted on.
    $connection.set(onHub)
    dropTilesForProfile('research', { connectionId: 'mini', profile: 'research' })

    expect(workspaceCwdForNewSession('research')).toBe('/srv/hub')
    expect(await desktop.profile.getDefault()).toEqual(hubDefault)
    expect($defaultProfileRoute.get()).toEqual(hubDefault)
  })

  it('opens a relayed chat in this tab only, not in a window that boots on this gateway', async () => {
    servingGateway()
    const desktop = install()
    await desktop.connections!.list()
    const open = vi.spyOn(window, 'open').mockReturnValue(null)

    const result = await desktop.openSessionWindow('m1', { connectionId: 'mini', profile: 'research' })

    expect(result.ok).toBe(false)
    expect(result.error).toMatch(/another gateway/)
    expect(open).not.toHaveBeenCalled()
  })

  it('lists the profiles of every saved gateway in the agent roster', async () => {
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

  it("lists a saved gateway's installed but unserved profile as unreachable and refuses its dial", async () => {
    // Mini serves only its own `default`; `research` is installed there but parked or standalone.
    servingGateway({ miniServes: ['default'] })
    const desktop = install()
    const roster = await desktop.getAgentRoster!()

    expect(
      roster.agents.filter(agent => agent.connectionId === 'mini').map(agent => [agent.profile, agent.reachable])
    ).toEqual([
      ['default', undefined],
      ['research', false]
    ])
    await expect(desktop.getGatewayWsUrlFor!({ connectionId: 'mini', profile: 'research' })).rejects.toBeInstanceOf(
      UnservedProfileError
    )
    await expect(desktop.getGatewayWsUrlFor!({ connectionId: 'mini', profile: 'default' })).resolves.toMatchObject({
      ok: true
    })
  })

  it('opens a profile created on a saved gateway since the last roster read', async () => {
    // Mini serves its `default`; `fresh` is created there, and served, after the roster was read.
    const miniProfiles = ['default', 'research']
    const miniServes = ['default']
    servingGateway({ miniProfiles, miniServes })
    const desktop = install()
    await desktop.getAgentRoster!()

    miniProfiles.push('fresh')
    miniServes.push('fresh')

    await expect(desktop.getGatewayWsUrlFor!({ connectionId: 'mini', profile: 'fresh' })).resolves.toMatchObject({
      ok: true
    })
    await expect(desktop.getGatewayWsUrlFor!({ connectionId: 'mini', profile: 'research' })).rejects.toBeInstanceOf(
      UnservedProfileError
    )
  })

  it("reads a re-pointed connection's served profiles afresh though it keeps its id", async () => {
    // Native Desktop edits Mini to point at another gateway, keeping the id `mini`. The new one has
    // the same profiles installed but runs as, and serves, `research` only.
    const mini = { own: 'default', row: MINI }
    const miniServes = ['default']
    servingGateway({ mini, miniServes })
    const desktop = install()
    await desktop.getAgentRoster!()

    mini.own = 'research'
    mini.row = { ...MINI, url: 'https://mini-2.example.ts.net' }
    miniServes.splice(0, miniServes.length)
    const roster = await desktop.getAgentRoster!()

    expect(
      roster.agents.filter(agent => agent.connectionId === 'mini').map(agent => [agent.profile, agent.reachable])
    ).toEqual([
      ['default', false],
      ['research', undefined]
    ])
    await expect(desktop.getGatewayWsUrlFor!({ connectionId: 'mini', profile: 'research' })).resolves.toMatchObject({
      ok: true
    })
  })

  it('lists a saved gateway that never answers as unreachable without holding the roster', async () => {
    servingGateway({ miniStalled: true })
    vi.useFakeTimers()
    let settled = false
    const pending = install().getAgentRoster!().finally(() => (settled = true))

    await vi.advanceTimersByTimeAsync(RELAYED_ROSTER_TIMEOUT_MS)
    expect(settled).toBe(true)

    const roster = await pending
    expect(roster.agents.map(agent => [agent.connectionId, agent.profile])).toEqual([['local', 'default']])
    expect(roster.sources.map(source => [source.connectionId, source.reachable])).toEqual([
      ['local', true],
      ['mini', false]
    ])
  })

  it('lists each saved gateway as its own roster source under its registry id and label', async () => {
    servingGateway()
    const roster = await install().getAgentRoster!()

    expect(roster.sources.map(source => [source.connectionId, source.kind, source.label])).toEqual([
      ['local', 'local', 'Hub'],
      ['mini', 'remote', 'Mini']
    ])
    expect(roster.agents.find(agent => agent.connectionId === 'mini')).toMatchObject({
      connectionKind: 'remote',
      connectionLabel: 'Mini'
    })
  })

  it('reaches only its own gateway where the host serves no saved connections', async () => {
    const requests = servingGateway()
    const desktop = install({ surface: 'dashboard' })

    expect(desktop.connections).toBeUndefined()
    await expect(desktop.getConnectionFor!({ connectionId: 'mini' })).rejects.toThrow(/mini/)
    expect(requests.some(request => request.path.includes('/relay/'))).toBe(false)
  })
})
