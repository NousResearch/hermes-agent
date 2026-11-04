import { afterEach, describe, expect, it, vi } from 'vitest'

import { $defaultProfileRoute, setDefaultProfile } from '@/store/default-profile'
import { $connection, workspaceCwdForNewSession } from '@/store/session'
import { dropTilesForProfile } from '@/store/session-states'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { dropBrowserProfilePrefs, migrateBrowserProfilePrefs } from './browser-profile-prefs'
import { RELAYED_ROSTER_TIMEOUT_MS } from './browser-relay'

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

/**
 * The serving gateway at /hermes, relaying to Mini; answers by path and records every request. A
 * stalled Mini takes the request and never answers it.
 */
function servingGateway({ miniStalled = false } = {}) {
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
      if (miniStalled) {
        return new Promise<Response>((_resolve, reject) =>
          init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
        )
      }

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
  $defaultProfileRoute.set(null)
  vi.useRealTimers()
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

  it("keeps the default folder a relayed gateway's profile chose apart from this gateway's", async () => {
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

  it('lists a relayed gateway that never answers as unreachable without holding the roster', async () => {
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

  it('reaches only its own gateway when the relay is off', async () => {
    const requests = servingGateway()
    const desktop = install({ relay: false })

    expect(desktop.gatewayRelay).toBeUndefined()
    await expect(desktop.getConnectionFor!({ connectionId: 'mini' })).rejects.toThrow(/mini/)
    expect(requests.some(request => request.path.includes('/relay/'))).toBe(false)
  })
})
