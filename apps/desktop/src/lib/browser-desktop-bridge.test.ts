import { afterEach, describe, expect, it, vi } from 'vitest'

import { reconnectMovedCloudAgent } from '@/app/settings/cloud-team-change'
import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { desktopGit } from './desktop-git'
import { openLink } from './external-link'
import { resolveDesktopGatewayWsUrl } from './gateway-ws-url'
import { downloadGatewayMediaFile, resolveMediaPlaybackSrc } from './media'

type MutableWindow = Window & {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_BASE_PATH__?: string
  __HERMES_DASHBOARD_PROFILE__?: string
  hermesDesktop?: Window['hermesDesktop']
}

const mutableWindow = () => window as unknown as MutableWindow

afterEach(() => {
  const win = mutableWindow()
  delete win.__HERMES_AUTH_REQUIRED__
  delete win.__HERMES_BASE_PATH__
  delete win.__HERMES_DASHBOARD_PROFILE__
  clearWebappSession()
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  Reflect.deleteProperty(navigator, 'clipboard')
  window.history.replaceState(null, '', '/#/')
  $connection.set(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('browser-hosted Desktop bridge', () => {
  it('does not make network requests to resolve link titles', async () => {
    seedWebappSession('served-token')
    const fetchMock = vi.fn().mockResolvedValue(new Response('<title>Untrusted title</title>'))
    vi.stubGlobal('fetch', fetchMock)
    expect(installBrowserDesktopBridge()).toBe(true)

    await expect(mutableWindow().hermesDesktop!.fetchLinkTitle('https://example.com/some-article')).resolves.toBe('')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('refuses cloud-team reconnection without a native connection registry', async () => {
    seedWebappSession('served-token')
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    expect(installBrowserDesktopBridge()).toBe(true)
    const desktop = mutableWindow().hermesDesktop!
    const logout = vi.spyOn(desktop, 'oauthLogoutConnectionConfig')
    const signIn = vi.spyOn(desktop.cloud, 'agentSignIn')

    await expect(reconnectMovedCloudAgent(desktop, {
      id: 'cloud-agent',
      kind: 'cloud',
      label: 'Cloud agent',
      url: 'https://agent.example.com',
      authMode: 'oauth',
      org: 'old-team',
      tokenSet: false,
      tokenPreview: null
    }, 'new-team', () => true)).resolves.toBe(false)
    expect(logout).not.toHaveBeenCalled()
    expect(signIn).not.toHaveBeenCalled()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('does not replace the Electron preload bridge', () => {
    const win = mutableWindow()
    const existing = { api: vi.fn() } as unknown as Window['hermesDesktop']
    seedWebappSession('served-token')
    win.hermesDesktop = existing

    expect(installBrowserDesktopBridge()).toBe(false)
    expect(win.hermesDesktop).toBe(existing)
  })

  it('requires the server-injected session token', () => {
    expect(installBrowserDesktopBridge()).toBe(false)
    expect(mutableWindow().hermesDesktop).toBeUndefined()
  })

  it('installs behind the authenticated non-loopback server gate without exposing a token', async () => {
    const win = mutableWindow()
    win.__HERMES_AUTH_REQUIRED__ = true
    win.__HERMES_BASE_PATH__ = '/hermes'

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        headers: { 'content-type': 'application/json' },
        status: 200
      })
    )

    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)
    await win.hermesDesktop!.api({ path: '/api/status' })

    const [requestUrl, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(requestUrl.pathname).toBe('/hermes/api/status')
    expect(init.credentials).toBe('same-origin')
    expect(new Headers(init.headers).has('X-Hermes-Session-Token')).toBe(false)

    const connection = await win.hermesDesktop!.getConnection()
    // The gateway store routes this connection's requests onto the window's socket by its id.
    expect(connection.connectionId).toBe('local')
    expect(connection.authMode).toBe('oauth')
    expect(connection.mode).toBe('remote')
    expect(connection.token).toBe('')
    // The serving gateway's canonical socket: no credential in the descriptor.
    const ws = new URL(connection.wsUrl)
    expect(ws.pathname).toBe('/hermes/api/ws')
    expect(ws.searchParams.get('native_dial')).toBe('browser')
    expect(ws.searchParams.has('ticket') || ws.searchParams.has('token')).toBe(false)
  })

  // About renders `Version <appVersion>` / `v<appVersion>`: it must carry the
  // serving host's release, never a host-kind label.
  it('reports the serving host version, unknown when the host cannot say', async () => {
    const win = mutableWindow()
    win.__HERMES_AUTH_REQUIRED__ = true
    win.__HERMES_BASE_PATH__ = '/hermes'

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ displayVersion: '0.21.5+1913', version: '0.21.5' })))
      .mockRejectedValueOnce(new Error('offline'))

    vi.stubGlobal('fetch', fetchMock)
    expect(installBrowserDesktopBridge()).toBe(true)

    await expect(win.hermesDesktop!.getVersion()).resolves.toMatchObject({ appVersion: '0.21.5+1913' })
    expect((fetchMock.mock.calls[0] as [URL])[0].pathname).toBe('/hermes/api/health')
    // '' is About's "version unavailable" state.
    await expect(win.hermesDesktop!.getVersion()).resolves.toMatchObject({ appVersion: '' })
  })

  it('mints a fresh gateway ticket for each WebSocket dial', async () => {
    const win = mutableWindow()
    win.__HERMES_AUTH_REQUIRED__ = true
    win.__HERMES_BASE_PATH__ = '/hermes'

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ ticket: 'ticket-one' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ticket: 'ticket-two' }), { status: 200 }))

    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)

    const first = await win.hermesDesktop!.getGatewayWsUrl('worker-a')
    const second = await win.hermesDesktop!.getGatewayWsUrlFor!({ connectionId: 'local', profile: 'worker-a' })

    const wsUrl = (result: typeof first) =>
      typeof result === 'string' ? result : result.ok ? result.wsUrl : ''

    const firstUrl = new URL(wsUrl(first))
    const secondUrl = new URL(wsUrl(second))
    expect(firstUrl.pathname).toBe('/hermes/api/ws')
    // native_dial: the client speaks the gateway's canonical contract and moves the ticket into
    // the socket's subprotocol.
    expect(firstUrl.searchParams.get('native_dial')).toBe('browser')
    expect(firstUrl.searchParams.get('ticket')).toBe('ticket-one')
    expect(firstUrl.searchParams.has('token')).toBe(false)
    expect(firstUrl.searchParams.get('profile')).toBe('worker-a')
    expect(secondUrl.searchParams.get('ticket')).toBe('ticket-two')
    expect(fetchMock).toHaveBeenCalledTimes(2)
    const minted = fetchMock.mock.calls.map(([url]) => url as URL)
    expect(minted.every(url => url.pathname === '/hermes/api/webapp/socket-ticket')).toBe(true)
    expect(minted.every(url => url.searchParams.get('socket') === 'gateway')).toBe(true)
    expect(minted.every(url => url.searchParams.get('profile') === 'worker-a')).toBe(true)
    expect(fetchMock.mock.calls.every(([, init]) => init?.method === 'POST')).toBe(true)
    expect(fetchMock.mock.calls.every(([, init]) => init?.credentials === 'same-origin')).toBe(true)
  })

  it('returns an actionable reauth result when the gated cookie expires', async () => {
    const win = mutableWindow()
    win.__HERMES_AUTH_REQUIRED__ = true
    win.__HERMES_BASE_PATH__ = '/hermes'
    let loginUrl = ''
    window.addEventListener(
      'hermes:browser-reauth-required',
      event => {
        event.preventDefault()
        loginUrl = String((event as CustomEvent<{ loginUrl?: string }>).detail.loginUrl || '')
      },
      { once: true }
    )
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            error: 'session_expired',
            login_url: '/hermes/login?next=%2Fapi%2Fauth%2Fws-ticket'
          }),
          { status: 401 }
        )
      )
    )

    expect(installBrowserDesktopBridge()).toBe(true)

    await expect(win.hermesDesktop!.getGatewayWsUrl('worker-a')).resolves.toMatchObject({
      error: expect.stringContaining('session_expired'),
      needsOauthLogin: true,
      ok: false
    })
    expect(new URL(loginUrl).pathname).toBe('/hermes/login')
  })

  it('signs out through the server before navigating to the gated login page', async () => {
    const win = mutableWindow()
    win.__HERMES_AUTH_REQUIRED__ = true
    win.__HERMES_BASE_PATH__ = '/hermes'
    let loginUrl = ''
    window.addEventListener(
      'hermes:browser-reauth-required',
      event => {
        event.preventDefault()
        loginUrl = String((event as CustomEvent<{ loginUrl?: string }>).detail.loginUrl || '')
      },
      { once: true }
    )
    const fetchMock = vi.fn().mockResolvedValue(new Response('', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)
    await expect(win.hermesDesktop!.oauthLogoutConnectionConfig(window.location.origin)).resolves.toEqual({
      connected: false,
      ok: true
    })

    const [requestUrl, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(requestUrl.pathname).toBe('/hermes/auth/logout')
    expect(init.method).toBe('POST')
    expect(init.credentials).toBe('same-origin')
    expect(new URL(loginUrl).pathname).toBe('/hermes/login')
  })

  it('does not advertise Electron-only capabilities in browser-host mode', () => {
    const win = mutableWindow()
    seedWebappSession('served-token')

    expect(installBrowserDesktopBridge()).toBe(true)
    expect(win.hermesDesktop?.openBrowserWindow).toBeUndefined()
    expect(win.hermesDesktop?.onBrowserPopoutClosed).toBeUndefined()
    expect(win.hermesDesktop?.getSecretStorageEncryption).toBeUndefined()
    expect(win.hermesDesktop?.setSecretStorageEncryption).toBeUndefined()
    expect(win.hermesDesktop?.openSessionInTerminal).toBeUndefined()
    expect(win.hermesDesktop?.connections).toBeUndefined()
    expect(win.hermesDesktop?.onOpenFindBarRequested).toBeUndefined()
    expect(win.hermesDesktop?.petOverlay).toBeUndefined()
    expect(win.hermesDesktop?.quickEntry).toBeUndefined()
  })

  it('disables native window controls and rejects their operations without side effects', () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    const fetchMock = vi.fn()
    const closeMock = vi.spyOn(window, 'close').mockImplementation(() => undefined)
    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)
    const controls = win.hermesDesktop!.windowControls
    expect(controls.custom).toBe(false)

    for (const operation of [controls.minimize, controls.toggleMaximize, controls.close]) {
      expect(operation).toThrow()
    }

    expect(fetchMock).not.toHaveBeenCalled()
    expect(closeMock).not.toHaveBeenCalled()
  })

  it('keeps clipboard writes on the browser-native method captured before renderer shims', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    const nativeWriteText = vi.fn().mockResolvedValue(undefined)
    const laterShim = vi.fn().mockRejectedValue(new Error('recursive clipboard shim'))
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: nativeWriteText }
    })

    expect(installBrowserDesktopBridge()).toBe(true)
    Object.defineProperty(navigator.clipboard, 'writeText', {
      configurable: true,
      value: laterShim
    })

    await expect(win.hermesDesktop!.writeClipboard('payload')).resolves.toBe(true)
    expect(nativeWriteText).toHaveBeenCalledWith('payload')
    expect(laterShim).not.toHaveBeenCalled()
  })

  it('remembers the selected browser profile in the launch URL without losing its route', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    window.history.replaceState(null, '', '/hermes?view=chat#/session-1')

    expect(installBrowserDesktopBridge()).toBe(true)
    await expect(win.hermesDesktop!.profile.remember(' research ')).resolves.toEqual({ profile: 'research' })

    let url = new URL(window.location.href)
    expect(url.pathname).toBe('/hermes')
    expect(url.searchParams.get('view')).toBe('chat')
    expect(url.searchParams.get('profile')).toBe('research')
    expect(url.hash).toBe('#/session-1')

    // Default is written out: a bare URL boots on the origin's saved default.
    await expect(win.hermesDesktop!.profile.remember('default')).resolves.toEqual({ profile: 'default' })
    url = new URL(window.location.href)
    expect(url.searchParams.get('profile')).toBe('default')
    expect(url.hash).toBe('#/session-1')

    await win.hermesDesktop!.profile.remember(null)
    expect(new URL(window.location.href).searchParams.has('profile')).toBe(false)
  })

  it('opens session windows on the owning profile HashRouter route with spectator flags before the hash', async () => {
    const win = mutableWindow()
    // A gated host opens the window directly; a launch-link tab hands it a session first.
    win.__HERMES_AUTH_REQUIRED__ = true
    window.history.replaceState(null, '', '/?profile=work#/existing')
    const open = vi.spyOn(window, 'open').mockImplementation(() => null)

    expect(installBrowserDesktopBridge()).toBe(true)
    await expect(
      win.hermesDesktop!.openSessionWindow('session / 1', { profile: 'research', watch: true })
    ).resolves.toEqual({ ok: true })

    const [rawUrl, target, features] = open.mock.calls[0]
    const url = new URL(String(rawUrl))
    expect(url.searchParams.get('profile')).toBe('research')
    expect(url.searchParams.get('win')).toBe('secondary')
    expect(url.searchParams.get('watch')).toBe('1')
    expect(url.hash).toBe('#/session%20%2F%201')
    expect(target).toBe('_blank')
    expect(features).toBe('noopener,noreferrer')
  })

  it('hands only the schemes Electron opens externally to window.open', async () => {
    seedWebappSession('served-token')
    const open = vi.spyOn(window, 'open').mockImplementation(() => null)

    expect(installBrowserDesktopBridge()).toBe(true)
    const desktop = mutableWindow().hermesDesktop!
    const openers = [desktop.openExternal, desktop.openPreviewInBrowser!]

    // Chat links reach the bridge through the same handoff as a user's click.
    openLink('javascript:alert(document.cookie)')

    for (const url of [
      'javascript:alert(1)',
      ' JavaScript:alert(1)',
      'data:text/html,<script>alert(1)</script>',
      'vbscript:msgbox(1)',
      'hermes-media://stream/%2Fetc%2Fpasswd',
      'file:///etc/passwd',
      'not a url'
    ]) {
      for (const opener of openers) {await opener(url)}
    }

    expect(open).not.toHaveBeenCalled()

    for (const url of ['https://example.com/a b', 'http://127.0.0.1:3000/', 'mailto:someone@example.com']) {
      for (const opener of openers) {await opener(url)}
    }

    expect(open.mock.calls).toEqual(
      ['https://example.com/a%20b', 'http://127.0.0.1:3000/', 'mailto:someone@example.com'].flatMap(url => [
        [url, '_blank', 'noopener,noreferrer'],
        [url, '_blank', 'noopener,noreferrer']
      ])
    )
  })

  it.each([
    { expectedProfile: 'work', name: 'absent', opts: undefined },
    { expectedProfile: 'work', name: 'undefined', opts: { profile: undefined } },
    { expectedProfile: 'work', name: 'null', opts: { profile: null } },
    { expectedProfile: 'work', name: 'blank', opts: { profile: '   ' } },
    { expectedProfile: 'default', name: 'default', opts: { profile: 'default' } },
    { expectedProfile: 'research', name: 'trimmed non-default', opts: { profile: ' research ' } }
  ])('resolves a $name session-window profile against the ambient launch URL', async ({ expectedProfile, opts }) => {
    const win = mutableWindow()
    // A gated host opens the window directly; a launch-link tab hands it a session first.
    win.__HERMES_AUTH_REQUIRED__ = true
    window.history.replaceState(null, '', '/?profile=work#/existing')
    const open = vi.spyOn(window, 'open').mockImplementation(() => null)

    expect(installBrowserDesktopBridge()).toBe(true)
    await expect(win.hermesDesktop!.openSessionWindow('session-1', opts)).resolves.toEqual({ ok: true })

    const [rawUrl] = open.mock.calls[0]
    expect(new URL(String(rawUrl)).searchParams.get('profile')).toBe(expectedProfile)
  })

  it('exposes one same-origin registry source for Bot profile routing', async () => {
    const win = mutableWindow()
    win.__HERMES_BASE_PATH__ = '/hermes'
    seedWebappSession('served-token')
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(async (url: URL) =>
        new Response(
          JSON.stringify(
            url.pathname.endsWith('/api/webapp/socket-ticket')
              ? { ticket: 'research-ticket' }
              : { profiles: [{ name: 'default' }, { name: 'research' }] }
          ),
          { status: 200 }
        )
      )
    )

    expect(installBrowserDesktopBridge()).toBe(true)

    await expect(win.hermesDesktop!.getProfileRoutes?.(['default', 'research', 'deleted'])).resolves.toEqual([
      { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' },
      { connectionId: 'local', mode: 'local', profile: 'research', targetProfile: 'research' }
    ])
    await expect(win.hermesDesktop!.getAgentRoster?.()).resolves.toMatchObject({
      agents: [
        { connectionId: 'local', connectionKind: 'local', handle: 'hermes', profile: 'default' },
        { connectionId: 'local', connectionKind: 'local', handle: 'research', profile: 'research' }
      ],
      primaryConnectionId: 'local',
      sources: [{ connectionId: 'local', kind: 'local', reachable: true }]
    })

    const connection = await win.hermesDesktop!.getConnectionFor?.({ connectionId: 'local', profile: 'research' })
    expect(connection).toMatchObject({
      connectionId: 'local',
      mode: 'remote',
      profile: 'research',
      registryScoped: true,
      sharedRemote: true
    })

    const result = await win.hermesDesktop!.getGatewayWsUrlFor?.({ connectionId: 'local', profile: 'research' })
    const wsUrl = typeof result === 'string' ? result : result?.ok ? result.wsUrl : ''
    expect(new URL(wsUrl).searchParams.get('profile')).toBe('research')
    await expect(
      win.hermesDesktop!.getConnectionFor?.({ connectionId: 'another-host', profile: 'research' })
    ).rejects.toThrow('No connection with id "another-host"')
  })

  function stubProfileStatus(status: Record<string, unknown>, profiles: { name: string }[]) {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(async (url: URL) => {
        const path = url.pathname

        const body = path.endsWith('/api/webapp/socket-ticket')
          ? { ticket: 'one-use' }
          : path.endsWith('/api/status')
            ? status
            : { profiles }

        return new Response(JSON.stringify(body), { status: 200 })
      })
    )
  }

  it('does not mark a profile this gateway does not serve as reachable', async () => {
    const win = mutableWindow()
    win.__HERMES_DASHBOARD_PROFILE__ = 'default'
    seedWebappSession('served-token')
    stubProfileStatus(
      { gateway_running: true, gateway_shared_with: ['default', 'research'] },
      [{ name: 'default' }, { name: 'research' }, { name: 'solo' }]
    )

    expect(installBrowserDesktopBridge()).toBe(true)
    const desktop = win.hermesDesktop!

    await expect(desktop.getProfileRoutes?.(['default', 'research', 'solo'])).resolves.toEqual([
      { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' },
      { connectionId: 'local', mode: 'local', profile: 'research', targetProfile: 'research' }
    ])

    const roster = await desktop.getAgentRoster?.()
    expect(roster?.sources).toEqual([
      expect.objectContaining({ connectionId: 'local', kind: 'local', reachable: true })
    ])
    expect(roster?.agents.find(agent => agent.profile === 'research')).toEqual(
      expect.not.objectContaining({ reachable: false })
    )
    expect(roster?.agents.find(agent => agent.profile === 'solo')).toMatchObject({
      connectionId: 'local',
      profile: 'solo',
      reachable: false
    })

    await expect(desktop.getConnection('research')).resolves.toMatchObject({
      profile: 'research',
      sharedPrimary: true
    })
    await expect(desktop.getConnection('solo')).resolves.toMatchObject({
      profile: 'solo',
      sharedPrimary: false
    })
    await expect(desktop.getConnectionFor?.({ connectionId: 'local', profile: 'research' })).resolves.toMatchObject({
      profile: 'research',
      sharedRemote: true
    })
    await expect(desktop.getConnectionFor?.({ connectionId: 'local', profile: 'solo' })).resolves.toMatchObject({
      profile: 'solo',
      sharedPrimary: false,
      sharedRemote: false
    })
  })

  it('treats an empty served list as this gateway serving only its own profile', async () => {
    const win = mutableWindow()
    win.__HERMES_DASHBOARD_PROFILE__ = 'default'
    seedWebappSession('served-token')
    stubProfileStatus(
      { gateway_running: true, gateway_shared_with: [] },
      [{ name: 'default' }, { name: 'research' }]
    )

    expect(installBrowserDesktopBridge()).toBe(true)
    const desktop = win.hermesDesktop!

    await expect(desktop.getProfileRoutes?.(['default', 'research'])).resolves.toEqual([
      { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' }
    ])
    await expect(desktop.getConnection('research')).resolves.toMatchObject({
      profile: 'research',
      sharedPrimary: false
    })
    await expect(desktop.getConnectionFor?.({ connectionId: 'local', profile: 'research' })).resolves.toMatchObject({
      sharedRemote: false
    })

    const roster = await desktop.getAgentRoster?.()
    expect(roster?.sources?.[0]?.reachable).toBe(true)
    expect(roster?.agents.find(agent => agent.profile === 'default')).toEqual(
      expect.not.objectContaining({ reachable: false })
    )
    expect(roster?.agents.find(agent => agent.profile === 'research')?.reachable).toBe(false)
  })

  it('keeps installed profiles reachable when the gateway does not report its served set', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    stubProfileStatus({ gateway_running: true, gateway_shared_with: null }, [
      { name: 'default' },
      { name: 'research' }
    ])

    expect(installBrowserDesktopBridge()).toBe(true)
    const desktop = win.hermesDesktop!

    await expect(desktop.getProfileRoutes?.(['default', 'research'])).resolves.toEqual([
      { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' },
      { connectionId: 'local', mode: 'local', profile: 'research', targetProfile: 'research' }
    ])
    await expect(desktop.getConnection('research')).resolves.toMatchObject({
      profile: 'research',
      sharedPrimary: true
    })
    await expect(desktop.getConnectionFor?.({ connectionId: 'local', profile: 'research' })).resolves.toMatchObject({
      sharedRemote: true
    })
    const roster = await desktop.getAgentRoster?.()
    expect(roster?.agents.every(agent => agent.reachable !== false)).toBe(true)
    expect(roster?.sources?.[0]?.reachable).toBe(true)
  })

  it('maps REST requests to the loopback API with token and profile scoping', async () => {
    const win = mutableWindow()
    win.__HERMES_BASE_PATH__ = '/hermes'
    seedWebappSession('served-token')

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        headers: { 'content-type': 'application/json' },
        status: 200
      })
    )

    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)

    await win.hermesDesktop!.api({
      path: '/api/config?view=desktop',
      profile: 'worker-a'
    })

    const [requestUrl, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(requestUrl.pathname).toBe('/hermes/api/config')
    expect(requestUrl.searchParams.get('view')).toBe('desktop')
    expect(requestUrl.searchParams.get('profile')).toBe('worker-a')
    expect(new Headers(init.headers).get('X-Hermes-Session-Token')).toBe('served-token')
  })

  it('routes Git through the browser transport and resolves the profile at call time', async () => {
    const win = mutableWindow()
    win.__HERMES_BASE_PATH__ = '/hermes'
    seedWebappSession('served-token')
    $connection.set({ profile: 'launch-profile' } as never)

    const worktrees = [{ branch: 'main', detached: false, isMain: true, locked: false, path: '/srv/my repo' }]

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ worktrees }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ diff: 'working diff' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }))

    vi.stubGlobal('fetch', fetchMock)
    expect(installBrowserDesktopBridge()).toBe(true)
    const git = desktopGit()!

    await expect(git.worktreeList('/srv/my repo')).resolves.toEqual(worktrees)
    const [listUrl, listInit] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(listUrl.pathname).toBe('/hermes/api/git/worktrees')
    expect(listUrl.searchParams.get('path')).toBe('/srv/my repo')
    expect(listUrl.searchParams.get('profile')).toBe('launch-profile')
    expect(new Headers(listInit.headers).get('X-Hermes-Session-Token')).toBe('served-token')

    $connection.set({ profile: 'active-profile' } as never)
    await expect(git.review.diff('/srv/my repo', 'a b.txt', 'uncommitted', null, false)).resolves.toBe(
      'working diff'
    )
    const [diffUrl] = fetchMock.mock.calls[1] as [URL, RequestInit]
    expect(diffUrl.searchParams.get('profile')).toBe('active-profile')
    expect(diffUrl.searchParams.get('file')).toBe('a b.txt')
    expect(diffUrl.searchParams.get('staged')).toBe('false')
    expect(diffUrl.searchParams.has('base')).toBe(false)

    await git.review.stage('/srv/my repo', '')
    const [stageUrl, stageInit] = fetchMock.mock.calls[2] as [URL, RequestInit]
    expect(stageUrl.pathname).toBe('/hermes/api/git/review/stage')
    expect(stageUrl.searchParams.get('profile')).toBe('active-profile')
    expect(stageInit.method).toBe('POST')
    expect(JSON.parse(String(stageInit.body))).toEqual({ file: null, path: '/srv/my repo' })

    const requestCount = fetchMock.mock.calls.length
    await expect(git.scanRepos(['/srv'])).resolves.toEqual([])
    expect(fetchMock).toHaveBeenCalledTimes(requestCount)
  })

  it('keeps recovery and filesystem bridge methods safe in browser mode', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    $connection.set({ profile: 'active-files' } as never)

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ path: '/tmp/example.txt', text: 'hello' }), { status: 200 })
    )

    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)

    expect(await win.hermesDesktop!.getRecentLogs()).toEqual({ lines: [], path: '' })
    expect((await win.hermesDesktop!.revealLogs()).ok).toBe(false)
    expect(await win.hermesDesktop!.readFileText('/tmp/example.txt')).toMatchObject({ text: 'hello' })

    const [requestUrl, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(requestUrl.pathname).toBe('/api/fs/read-text')
    expect(requestUrl.searchParams.get('path')).toBe('/tmp/example.txt')
    expect(requestUrl.searchParams.get('profile')).toBe('active-files')
    expect(new Headers(init.headers).get('X-Hermes-Session-Token')).toBe('served-token')
  })

  // Downloads and media elements cannot send the session header, and the
  // session token grants host shells: it must never appear in their URLs
  // (download history, "Copy video address"). Token sessions carry a ticket
  // minted for that exact route and query instead.
  const fileTicketFetch = (headStatus = 200) =>
    vi.fn(async (input: URL, init?: RequestInit) => {
      if (input.pathname.endsWith('/api/files/ticket')) {
        const { route, ...query } = JSON.parse(String(init?.body)) as Record<string, string>
        const scope = new URLSearchParams(query)
        scope.sort()

        return new Response(JSON.stringify({ ticket: `ticket:${route}:${scope}` }), { status: 200 })
      }

      return new Response(null, { status: headStatus })
    })

  const ticketFor = (url: URL, route: string) => {
    const query = new URLSearchParams(url.search)
    query.delete('ticket')
    query.sort()

    return `ticket:${route}:${query}`
  }

  const carriesToken = (url: URL, token: string) =>
    Boolean(token) && [...url.searchParams.entries()].some(entry => entry.join('=').includes(token))

  it.each(['token', 'cookie'])('downloads browser files with %s auth on the original profile', async auth => {
    const win = mutableWindow()
    const token = auth === 'token' ? 'served / token' : ''
    win.__HERMES_BASE_PATH__ = '/hermes'

    if (auth === 'cookie') {
      win.__HERMES_AUTH_REQUIRED__ = true
    } else {
      seedWebappSession(token)
    }

    const fetchMock = fileTicketFetch()
    vi.stubGlobal('fetch', fetchMock)
    const downloads: HTMLAnchorElement[] = []
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      downloads.push(this)
    })

    expect(installBrowserDesktopBridge()).toBe(true)
    $connection.set(await win.hermesDesktop!.getConnectionFor!({ connectionId: 'local', profile: 'research' }))

    const pending = downloadGatewayMediaFile('file:///srv/reports/a%20b.pdf', {
      sessionId: 'origin-session', profile: 'research'
    })

    $connection.set({ mode: 'remote', profile: 'switched-profile' } as never)
    await expect(pending).resolves.toMatchObject({ saved: true })

    const downloadCall = (fetchMock.mock.calls as [URL, RequestInit][]).find(([input]) =>
      input.pathname.endsWith('/api/fs/download')
    )

    expect(downloadCall).toBeTruthy()
    const [requestUrl, init] = downloadCall!
    // The workspace download route native Desktop uses: it reads an SSH-backed profile's files on
    // that machine, like the file tree that listed them, not a same-named file on the gateway host.
    expect(requestUrl.pathname).toBe('/hermes/api/fs/download')
    expect(requestUrl.searchParams.get('path')).toBe('file:///srv/reports/a%20b.pdf')
    expect(requestUrl.searchParams.get('profile')).toBe('research')
    expect(requestUrl.searchParams.get('session_id')).toBe('origin-session')
    expect(init.method).toBe('HEAD')
    expect(init.credentials).toBe('same-origin')
    expect(new Headers(init.headers).get('X-Hermes-Session-Token')).toBe(token || null)
    expect(downloads).toHaveLength(1)
    const url = new URL(downloads[0].href)
    expect(url.origin).toBe(window.location.origin)
    expect(url.pathname).toBe(requestUrl.pathname)
    expect(url.searchParams.get('path')).toBe(requestUrl.searchParams.get('path'))
    expect(url.searchParams.get('profile')).toBe('research')
    expect(url.searchParams.get('session_id')).toBe('origin-session')
    expect(carriesToken(url, token)).toBe(false)
    expect(url.searchParams.get('ticket')).toBe(token ? ticketFor(url, 'download') : null)
    expect(downloads[0].download).toBe('a b.pdf')
    expect(downloads[0].isConnected).toBe(false)

    fetchMock.mockImplementation(async () => new Response(null, { status: 404 }))
    await expect(downloadGatewayMediaFile('/srv/missing.pdf')).rejects.toThrow('404')
    await expect(win.hermesDesktop!.saveGatewayFile!({
      connectionId: 'another-host', path: '/srv/reports/a b.pdf'
    })).rejects.toThrow('No connection with id "another-host"')
    expect(downloads).toHaveLength(1)
  })

  it.each(['token', 'cookie'])('streams browser audio/video over HTTP with %s auth', async auth => {
    const win = mutableWindow()
    const token = auth === 'token' ? 'served / token' : ''
    win.__HERMES_BASE_PATH__ = '/hermes'

    if (auth === 'cookie') {
      win.__HERMES_AUTH_REQUIRED__ = true
    } else {
      seedWebappSession(token)
    }

    const fetchMock = fileTicketFetch()
    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)
    $connection.set(await win.hermesDesktop!.getConnectionFor!({ connectionId: 'local', profile: 'research' }))

    for (const path of [
      '/srv/a b.mp4',
      'file:///srv/voice%20memo.m4a',
      'C:/Users/Alice/video.mp4',
      'file:///C:/Users/Alice/video%20clip.mp4',
      'file://nas/share/video%20clip.mp4'
    ]) {
      const url = new URL(await resolveMediaPlaybackSrc(path))
      expect(url.origin).toBe(window.location.origin)
      expect(url.pathname).toBe('/hermes/api/files/stream')
      expect.soft(url.searchParams.get('path'), path).toBe(path)
      expect(url.searchParams.get('profile')).toBe('research')
      expect(carriesToken(url, token)).toBe(false)
      expect(url.searchParams.get('ticket')).toBe(token ? ticketFor(url, 'stream') : null)
    }

    await expect(resolveMediaPlaybackSrc('https://cdn.example.com/video.mp4')).resolves.toBe(
      'https://cdn.example.com/video.mp4'
    )

    const ticketCalls = (fetchMock.mock.calls as [URL, RequestInit][]).filter(([input]) =>
      input.pathname.endsWith('/api/files/ticket')
    )

    for (const [input, init] of ticketCalls) {
      expect(input.pathname).toBe('/hermes/api/files/ticket')
      expect(new Headers(init.headers).get('X-Hermes-Session-Token')).toBe(token)
    }

    expect(ticketCalls).toHaveLength(token ? 5 : 0)
  })

  it('persists browser image bytes through the existing chat upload API', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    $connection.set({ profile: 'active-images' } as never)

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ path: '/srv/hermes/images/upload.png' }), {
        status: 200
      })
    )

    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)

    const path = await win.hermesDesktop!.saveImageBuffer(
      new Uint8Array([0x89, 0x50, 0x4e, 0x47]),
      '.png'
    )

    expect(path).toBe('/srv/hermes/images/upload.png')
    const [requestUrl, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(requestUrl.pathname).toBe('/api/chat/image-upload')
    expect(requestUrl.searchParams.get('profile')).toBe('active-images')
    const payload = JSON.parse(String(init.body)) as { data_url: string; filename: string }
    expect(payload.filename).toBe('desktop-upload.png')
    expect(payload.data_url).toBe('data:image/png;base64,iVBORw==')
  })

  it('stages user-selected browser files on the host before returning paths', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    $connection.set({ profile: 'active-profile' } as never)

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ path: '/srv/hermes/uploads/notes.txt' }), {
        headers: { 'content-type': 'application/json' },
        status: 200
      })
    )

    vi.stubGlobal('fetch', fetchMock)
    vi.spyOn(HTMLInputElement.prototype, 'click').mockImplementation(function (this: HTMLInputElement) {
      Object.defineProperty(this, 'files', {
        configurable: true,
        value: [new File(['hello'], 'notes.txt', { type: 'text/plain' })]
      })
      this.dispatchEvent(new Event('change'))
    })

    expect(installBrowserDesktopBridge()).toBe(true)

    const paths = await win.hermesDesktop!.selectPaths({
      filters: [{ extensions: ['txt'], name: 'Text' }],
      multiple: true,
      profile: 'explicit-file-profile'
    })

    expect(paths).toEqual(['/srv/hermes/uploads/notes.txt'])
    const [requestUrl, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(requestUrl.pathname).toBe('/api/chat/file-upload')
    expect(requestUrl.searchParams.get('profile')).toBe('explicit-file-profile')
    expect(init.method).toBe('POST')
    expect(init.credentials).toBe('same-origin')
    expect(new Headers(init.headers).get('X-Hermes-Session-Token')).toBe('served-token')
    expect((init.body as FormData).get('file')).toBeInstanceOf(File)
    expect(document.querySelector('input[type="file"]')).toBeNull()
  })

  it('stages dropped browser file bytes before returning a host attachment path', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    $connection.set({ profile: 'drop-profile' } as never)

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ path: '/srv/hermes/uploads/drop.txt' }), { status: 200 })
    )

    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)
    const file = new File(['dropped'], 'drop.txt', { type: 'text/plain' })

    await expect(win.hermesDesktop!.stageFileForAttach?.(file)).resolves.toBe('/srv/hermes/uploads/drop.txt')
    const [requestUrl, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(requestUrl.pathname).toBe('/api/chat/file-upload')
    expect(requestUrl.searchParams.get('profile')).toBe('drop-profile')
    const stagedFile = (init.body as FormData).get('file') as File
    expect(stagedFile).toBeInstanceOf(File)
    expect(stagedFile.name).toBe('drop.txt')
    expect(stagedFile.size).toBe(file.size)
    expect(stagedFile.type).toBe('text/plain')
  })

  it('opens a 2 MB HTML page through srcdoc, byte-exact and inert to the wrapper, never through a frame URL', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    const blobs: Blob[] = []

    vi.spyOn(URL, 'createObjectURL').mockImplementation(blob => {
      blobs.push(blob as Blob)

      return 'blob:http://127.0.0.1:9119/large'
    })

    expect(installBrowserDesktopBridge()).toBe(true)
    // Chromium will not navigate a frame to a URL over 2 MiB: the old data: carrier left this blank.
    const page = `<!doctype html><p title="a&amp;b">"q" &lt;x&gt;</p><!-- "></iframe><script>parent.x=1</script> --><script>/*${'x'.repeat(2_000_000)}*/</script>`
    await win.hermesDesktop!.saveImageBuffer(new TextEncoder().encode(page), '.html')
    const wrapper = new DOMParser().parseFromString(await blobs[0]!.text(), 'text/html')
    const frame = wrapper.querySelector('iframe')!

    expect(wrapper.querySelectorAll('iframe, script')).toHaveLength(1)
    expect(frame.getAttribute('sandbox')).toBe('allow-scripts')
    expect(frame.hasAttribute('src')).toBe(false)
    expect(frame.getAttribute('srcdoc')).toBe(
      page.replace('<!doctype html>', '<!doctype html><base href="about:srcdoc">')
    )
  })

  it('gives an HTML page without its own <base> an about:srcdoc base so #anchor links stay in the page', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    const blobs: Blob[] = []

    vi.spyOn(URL, 'createObjectURL').mockImplementation(blob => {
      blobs.push(blob as Blob)

      return `blob:http://127.0.0.1:9119/page-${blobs.length}`
    })

    expect(installBrowserDesktopBridge()).toBe(true)

    const srcdocOf = async (page: string) => {
      await win.hermesDesktop!.saveImageBuffer(new TextEncoder().encode(page), '.html')
      const wrapper = new DOMParser().parseFromString(await blobs.at(-1)!.text(), 'text/html')

      return wrapper.querySelector('iframe')!.getAttribute('srcdoc')!
    }

    // Without a base of its own, the page inherits the wrapper's blob: URL and href="#sec" goes nowhere.
    const bare = '<h1><a href="#sec">jump</a></h1><h2 id="sec">sec</h2>'
    expect(await srcdocOf(bare)).toBe(`<base href="about:srcdoc">${bare}`)

    // The base goes after the doctype, so the page is not pushed into quirks mode.
    const standards = await srcdocOf(`<!DOCTYPE html>${bare}`)
    expect(standards).toBe(`<!DOCTYPE html><base href="about:srcdoc">${bare}`)
    const parsed = new DOMParser().parseFromString(standards, 'text/html')
    expect(parsed.compatMode).toBe('CSS1Compat')
    expect(parsed.querySelectorAll('base')).toHaveLength(1)
    expect(parsed.querySelector('base')!.getAttribute('href')).toBe('about:srcdoc')

    // A page that sets its own base keeps it, untouched.
    const based = '<html><head><BASE href="https://example.com/docs/"></head><body><a href="#sec">x</a></body></html>'
    expect(await srcdocOf(based)).toBe(based)
  })

  it('puts the about:srcdoc base inside the page\'s own <head>, so the head keeps its attributes', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    const blobs: Blob[] = []

    vi.spyOn(URL, 'createObjectURL').mockImplementation(blob => {
      blobs.push(blob as Blob)

      return `blob:http://127.0.0.1:9119/page-${blobs.length}`
    })

    expect(installBrowserDesktopBridge()).toBe(true)

    const parsedSrcdocOf = async (page: string) => {
      await win.hermesDesktop!.saveImageBuffer(new TextEncoder().encode(page), '.html')
      const wrapper = new DOMParser().parseFromString(await blobs.at(-1)!.text(), 'text/html')
      const srcdoc = wrapper.querySelector('iframe')!.getAttribute('srcdoc')!

      return { doc: new DOMParser().parseFromString(srcdoc, 'text/html'), srcdoc }
    }

    // A base before <head> would open an implicit head, and the authored tag's attributes would be lost.
    const report = '<!DOCTYPE html><HTML lang="en"><Head id="report-assets" data-x="1"><title>r</title></Head><body><a href="#sec">x</a></body></HTML>'
    const withHead = await parsedSrcdocOf(report)
    expect(withHead.srcdoc).toBe(report.replace('data-x="1">', 'data-x="1"><base href="about:srcdoc">'))
    expect(withHead.doc.compatMode).toBe('CSS1Compat')
    expect(withHead.doc.head.id).toBe('report-assets')
    expect(withHead.doc.head.dataset.x).toBe('1')
    expect(withHead.doc.querySelectorAll('base')).toHaveLength(1)
    expect(withHead.doc.querySelector('base')!.parentElement).toBe(withHead.doc.head)
    expect(withHead.doc.querySelector('base')!.getAttribute('href')).toBe('about:srcdoc')

    // Without a <head>, the base goes right after <html>, which keeps its attributes too.
    const headless = '<!doctype html><html lang="fr" data-theme="dark"><body><p>x</p></body></html>'
    const withHtml = await parsedSrcdocOf(headless)
    expect(withHtml.srcdoc).toBe(headless.replace('data-theme="dark">', 'data-theme="dark"><base href="about:srcdoc">'))
    expect(withHtml.doc.documentElement.lang).toBe('fr')
    expect(withHtml.doc.documentElement.dataset.theme).toBe('dark')
    expect(withHtml.doc.querySelectorAll('base')).toHaveLength(1)
    expect(withHtml.doc.querySelector('base')!.parentElement).toBe(withHtml.doc.head)

    // <header> is not a <head>: the base stays after the doctype.
    const banner = '<!doctype html><header id="top"><h1>t</h1></header><p>x</p>'
    const withHeader = await parsedSrcdocOf(banner)
    expect(withHeader.srcdoc).toBe(banner.replace('<!doctype html>', '<!doctype html><base href="about:srcdoc">'))
    expect(withHeader.doc.querySelectorAll('base')).toHaveLength(1)
    expect(withHeader.doc.querySelector('header')!.id).toBe('top')
  })

  it('stages non-image buffers as browser-local object URLs', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    const createObjectURL = vi.fn((_blob: Blob | MediaSource) => 'blob:http://127.0.0.1:9119/preview')
    vi.spyOn(URL, 'createObjectURL').mockImplementation(createObjectURL)
    const revoke = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    const open = vi.spyOn(window, 'open').mockReturnValue(null)

    expect(installBrowserDesktopBridge()).toBe(true)

    const staged = await win.hermesDesktop!.saveImageBuffer(
      new TextEncoder().encode('<h1>preview</h1>'),
      '.html'
    )

    expect(staged).toBe('blob:http://127.0.0.1:9119/preview')
    expect(createObjectURL).toHaveBeenCalledTimes(1)
    const [blob] = createObjectURL.mock.calls[0]
    expect(blob).toBeInstanceOf(Blob)
    const wrapper = await (blob as Blob).text()

    expect(wrapper).toContain('sandbox="allow-scripts"')
    expect(wrapper).not.toContain('allow-same-origin')
    expect(wrapper).not.toContain('<h1>preview</h1>')
    expect(wrapper).toContain('srcdoc="&lt;base href=&quot;about:srcdoc&quot;&gt;&lt;h1&gt;preview&lt;/h1&gt;"')
    expect(wrapper).not.toMatch(/\ssrc="data:/)

    await win.hermesDesktop!.openPreviewInBrowser!(staged)
    expect(open).toHaveBeenCalledExactlyOnceWith(staged, '_blank', 'noopener,noreferrer')
    open.mockClear()
    await win.hermesDesktop!.openPreviewInBrowser!('blob:http://127.0.0.1:9119/unowned')
    await win.hermesDesktop!.openExternal(staged)
    expect(open).not.toHaveBeenCalled()
    window.dispatchEvent(new Event('beforeunload'))
    expect(revoke).toHaveBeenCalledWith(staged)
    await win.hermesDesktop!.openPreviewInBrowser!(staged)
    expect(open).not.toHaveBeenCalled()
  })

  it('rejects API paths that escape the Webapp origin', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)

    await expect(win.hermesDesktop!.api?.({ path: '//attacker.example/collect' })).rejects.toThrow(
      'must remain on the Webapp origin'
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('rejects API paths that escape the configured base path', async () => {
    const win = mutableWindow()
    win.__HERMES_BASE_PATH__ = '/hermes'
    seedWebappSession('served-token')
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)

    await expect(win.hermesDesktop!.api?.({ path: '/../collect' })).rejects.toThrow(
      'must remain inside the configured base path'
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('maps the browser-hosted terminal rail onto an authenticated host-shell PTY socket', async () => {
    const win = mutableWindow()
    win.__HERMES_BASE_PATH__ = '/hermes'
    seedWebappSession('served-token')
    $connection.set({ profile: 'active-terminal' } as never)

    const fetchMock = vi.fn().mockImplementation(async () => new Response(JSON.stringify({ ticket: 'shell-ticket' })))
    vi.stubGlobal('fetch', fetchMock)

    class FakeWebSocket {
      static readonly CONNECTING = 0
      static readonly OPEN = 1
      static readonly CLOSING = 2
      static readonly CLOSED = 3
      static instances: FakeWebSocket[] = []
      binaryType = ''
      onclose: ((event: CloseEvent) => void) | null = null
      onerror: (() => void) | null = null
      onmessage: ((event: MessageEvent) => void) | null = null
      onopen: (() => void) | null = null
      readyState = FakeWebSocket.CONNECTING
      sent: string[] = []
      readonly url: string

      constructor(url: string | URL) {
        this.url = String(url)
        FakeWebSocket.instances.push(this)
        queueMicrotask(() => {
          this.readyState = FakeWebSocket.OPEN
          this.onopen?.()
          this.onmessage?.({
            data: '\u0000HERMES_TERMINAL_META:{"shell":"powershell.exe"}'
          } as MessageEvent)
        })
      }

      close(code = 1000, reason = '') {
        this.readyState = FakeWebSocket.CLOSED
        this.onclose?.(new CloseEvent('close', { code, reason }))
      }

      send(data: string) {
        this.sent.push(data)
      }

      emitBytes(text: string) {
        this.emitRaw(new TextEncoder().encode(text))
      }

      emitRaw(bytes: Uint8Array) {
        this.onmessage?.({ data: bytes.buffer } as MessageEvent)
      }
    }

    vi.stubGlobal('WebSocket', FakeWebSocket)
    expect(installBrowserDesktopBridge()).toBe(true)

    const session = await win.hermesDesktop!.terminal.start({ cols: 42, cwd: '/work', rows: 13 })
    const socket = FakeWebSocket.instances[0]!
    const url = new URL(socket.url)
    expect(session).toMatchObject({ cwd: '/work', shell: 'powershell.exe' })
    await expect(win.hermesDesktop!.terminal.attach(session.id)).resolves.toBe(true)
    await expect(win.hermesDesktop!.terminal.attach('missing-terminal')).resolves.toBe(false)
    expect(url.pathname).toBe('/hermes/api/host-terminal')
    expect(url.searchParams.get('ticket')).toBe('shell-ticket')
    expect(socket.url).not.toContain('served-token')
    expect((fetchMock.mock.calls[0][0] as URL).pathname).toBe('/hermes/api/webapp/socket-ticket')
    expect((fetchMock.mock.calls[0][0] as URL).searchParams.get('socket')).toBe('host-terminal')
    expect(url.searchParams.get('profile')).toBe('active-terminal')
    expect(url.searchParams.get('cwd')).toBe('/work')
    expect(url.searchParams.get('cols')).toBe('42')
    expect(url.searchParams.get('rows')).toBe('13')

    const output = vi.fn()
    const exited = vi.fn()
    const stopData = win.hermesDesktop!.terminal.onData(session.id, output)
    const stopExit = win.hermesDesktop!.terminal.onExit(session.id, exited)
    socket.emitBytes('hello from host shell')
    await Promise.resolve()
    expect(output).toHaveBeenCalledWith('hello from host shell')

    const multibyte = new TextEncoder().encode('😀')
    socket.emitRaw(multibyte.slice(0, 2))
    socket.emitRaw(multibyte.slice(2))
    expect(output).toHaveBeenCalledWith('😀')
    expect(output).not.toHaveBeenCalledWith(expect.stringContaining('�'))

    await expect(win.hermesDesktop!.terminal.write(session.id, 'hello\r')).resolves.toBe(true)
    await expect(win.hermesDesktop!.terminal.resize(session.id, { cols: 80, rows: 24 })).resolves.toBe(true)
    expect(socket.sent).toEqual(['hello\r', '\u001b[RESIZE:80;24]'])
    await expect(win.hermesDesktop!.terminal.cwd(session.id)).resolves.toBeNull()

    socket.close(4410, 'shell exited')
    expect(exited).toHaveBeenCalledWith({ code: null, signal: null })
    stopData()
    stopExit()
    await expect(win.hermesDesktop!.terminal.dispose(session.id)).resolves.toBe(true)
  })

  it('opens the host shell of a signed-in browser with a terminal ticket from its cookie session', async () => {
    const win = mutableWindow()
    win.__HERMES_AUTH_REQUIRED__ = true
    const fetchMock = vi.fn().mockImplementation(async () => new Response(JSON.stringify({ ticket: 'shell-ticket' })))
    vi.stubGlobal('fetch', fetchMock)
    const dialed: string[] = []

    class SignedInSocket {
      binaryType = ''
      onclose: ((event: CloseEvent) => void) | null = null
      onerror: (() => void) | null = null
      onmessage: ((event: MessageEvent) => void) | null = null
      onopen: (() => void) | null = null
      readyState = 0

      constructor(url: string | URL) {
        dialed.push(String(url))
        queueMicrotask(() => {
          this.readyState = 1
          this.onopen?.()
          this.onmessage?.({ data: '\u0000HERMES_TERMINAL_META:{"shell":"sh"}' } as MessageEvent)
        })
      }

      close() {}
      send() {}
    }

    vi.stubGlobal('WebSocket', SignedInSocket)
    expect(installBrowserDesktopBridge()).toBe(true)
    await win.hermesDesktop!.terminal.start({ cwd: '/work' })

    const [mint, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(mint.pathname).toBe('/api/webapp/socket-ticket')
    expect(mint.searchParams.get('socket')).toBe('host-terminal')
    expect(init.method).toBe('POST')
    expect(init.credentials).toBe('same-origin')
    expect(new Headers(init.headers).has('X-Hermes-Session-Token')).toBe(false)
    expect(new URL(dialed[0]).searchParams.get('ticket')).toBe('shell-ticket')
  })

  it('buffers early host-shell output until Desktop registers its terminal data listener', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async () => new Response(JSON.stringify({ ticket: 'shell-ticket' }))))

    class PromptWebSocket {
      static readonly OPEN = 1
      binaryType = ''
      onclose: ((event: CloseEvent) => void) | null = null
      onerror: (() => void) | null = null
      onmessage: ((event: MessageEvent) => void) | null = null
      onopen: (() => void) | null = null
      readyState = 0

      constructor(_url: string | URL) {
        queueMicrotask(() => {
          this.readyState = PromptWebSocket.OPEN
          this.onopen?.()
          this.onmessage?.({
            data: '\u0000HERMES_TERMINAL_META:{"shell":"fish"}'
          } as MessageEvent)
          const bytes = new TextEncoder().encode('host shell ready')
          this.onmessage?.({ data: bytes.buffer } as MessageEvent)
        })
      }

      close() {
        this.readyState = 3
      }

      send(_data: string) {}
    }

    vi.stubGlobal('WebSocket', PromptWebSocket)
    expect(installBrowserDesktopBridge()).toBe(true)
    const session = await win.hermesDesktop!.terminal.start({ cwd: '/home' })
    expect(session.shell).toBe('fish')
    const output = vi.fn()
    win.hermesDesktop!.terminal.onData(session.id, output)
    await Promise.resolve()
    expect(output).toHaveBeenCalledWith('host shell ready')
  })

  it('exposes a profile-scoped gateway URL for Desktop boot', async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')

    expect(installBrowserDesktopBridge()).toBe(true)

    const connection = await win.hermesDesktop!.getConnection('worker-a')
    const ws = new URL(connection.wsUrl)
    expect(connection.mode).toBe('remote')
    expect(connection.profile).toBe('worker-a')
    expect(ws.pathname).toBe('/api/ws')
    expect(ws.searchParams.get('native_dial')).toBe('browser')
    // The session token never rides the socket URL: every dial mints its own one-use ticket.
    expect(ws.searchParams.has('token')).toBe(false)
    expect(ws.searchParams.get('profile')).toBe('worker-a')
  })

  // A tab's own socket answers for the profile the serving gateway was launched as whenever a request
  // names none, as its REST does. So that is the tab's own connection whatever `?profile=` the tab is on;
  // every other profile is reached on that socket with `profile` on each request (sharedPrimary), the
  // route a live switch takes.
  it.each([
    { launch: 'default', other: 'research' },
    { launch: 'work', other: 'default' }
  ])("keeps the $launch launch profile as the tab's own connection on ?profile=$other", async ({ launch, other }) => {
    const win = mutableWindow()
    win.__HERMES_DASHBOARD_PROFILE__ = launch
    seedWebappSession('served-token')
    window.history.replaceState(null, '', `/?profile=${other}#/`)
    const fetchMock = vi.fn(async (_url: URL) => new Response(JSON.stringify({ ticket: 'one-use' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)
    const desktop = win.hermesDesktop!

    const own = await desktop.getConnection()
    expect(own.profile).toBeUndefined()
    expect(own.sharedPrimary).toBeFalsy()
    expect(new URL(own.wsUrl).searchParams.has('profile')).toBe(false)
    await resolveDesktopGatewayWsUrl(desktop, own)
    const mint = new URL(String(fetchMock.mock.calls[0][0]))
    expect(mint.pathname).toBe('/api/webapp/socket-ticket')
    expect(mint.searchParams.get('socket')).toBe('gateway')
    expect(mint.searchParams.has('profile')).toBe(false)

    await expect(desktop.getConnection(other)).resolves.toMatchObject({ profile: other, sharedPrimary: true })
    await expect(desktop.getConnection(launch)).resolves.toMatchObject({ profile: launch, sharedPrimary: false })
  })

  // The serving gateway mints the ticket itself, so there is no gateway to start and no verdict to
  // translate: a failed mint is reported as the route answered it.
  it("reports a failed mint as the route's own answer", async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('{"detail":"gateway unavailable"}', { status: 503 }))
    )

    expect(installBrowserDesktopBridge()).toBe(true)

    await expect(win.hermesDesktop!.getGatewayWsUrl('worker-a')).rejects.toThrow(
      '503: {"detail":"gateway unavailable"}'
    )
  })

  it("rejects a dial whose ticket mint fails instead of dialing the Webapp's credential-free socket URL", async () => {
    const win = mutableWindow()
    seedWebappSession('served-token')
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(
        () => new Response('{"detail":"gateway unavailable"}', { status: 503 })
      )
    )
    expect(installBrowserDesktopBridge()).toBe(true)
    const desktop = win.hermesDesktop!
    const connection = await desktop.getConnection('worker-a')

    await expect(resolveDesktopGatewayWsUrl(desktop, connection)).rejects.toThrow('503')
  })

  it('bounds the ticket mint by the ordinary request timeout, not a gateway cold start', async () => {
    vi.useFakeTimers()

    try {
      const win = mutableWindow()
      seedWebappSession('served-token')
      vi.stubGlobal(
        'fetch',
        vi.fn().mockImplementation(
          (_url: URL, init: RequestInit) =>
            new Promise((_resolve, reject) =>
              init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
            )
        )
      )

      expect(installBrowserDesktopBridge()).toBe(true)

      const minted = win.hermesDesktop!.getGatewayWsUrl('worker-a')
      const settled = expect(minted).rejects.toThrow('aborted')
      await vi.advanceTimersByTimeAsync(30_000)
      await settled
    } finally {
      vi.useRealTimers()
    }
  })
})
