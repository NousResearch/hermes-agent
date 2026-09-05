import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { downloadGatewayMediaFile, resolveMediaPlaybackSrc } from './media'

type MutableWindow = Window & {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_BASE_PATH__?: string
  __HERMES_SESSION_TOKEN__?: string
  hermesDesktop?: Window['hermesDesktop']
}

const mutableWindow = () => window as unknown as MutableWindow

afterEach(() => {
  const win = mutableWindow()
  delete win.__HERMES_AUTH_REQUIRED__
  delete win.__HERMES_BASE_PATH__
  delete win.__HERMES_SESSION_TOKEN__
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  Reflect.deleteProperty(navigator, 'clipboard')
  window.history.replaceState(null, '', '/#/')
  $connection.set(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('browser-hosted Desktop bridge', () => {
  it('does not replace the Electron preload bridge', () => {
    const win = mutableWindow()
    const existing = { api: vi.fn() } as unknown as Window['hermesDesktop']
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
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
    expect(connection.authMode).toBe('oauth')
    expect(connection.mode).toBe('remote')
    expect(connection.token).toBe('')
    expect(connection.wsUrl).toBe('')
  })

  it('mints a fresh single-use ticket for each gated WebSocket URL', async () => {
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
    expect(firstUrl.searchParams.get('ticket')).toBe('ticket-one')
    expect(firstUrl.searchParams.has('token')).toBe(false)
    expect(firstUrl.searchParams.get('profile')).toBe('worker-a')
    expect(secondUrl.searchParams.get('ticket')).toBe('ticket-two')
    expect(fetchMock).toHaveBeenCalledTimes(2)
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
    win.__HERMES_SESSION_TOKEN__ = 'served-token'

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

  it.each(['getPoolLimits', 'setPoolLimits'])(
    'rejects native backend pool operations through %s without a server request',
    async method => {
      const win = mutableWindow()
      win.__HERMES_SESSION_TOKEN__ = 'served-token'
      const fetchMock = vi.fn()
      vi.stubGlobal('fetch', fetchMock)

      expect(installBrowserDesktopBridge()).toBe(true)
      const operation = Reflect.get(win.hermesDesktop!, method)
      expect(typeof operation).toBe('function')

      await expect(operation({ maxBackends: 5 })).rejects.toThrow(
        'Desktop backend pool sizing is not available in the browser-hosted Desktop'
      )
      expect(fetchMock).not.toHaveBeenCalled()
    }
  )

  it('keeps clipboard writes on the browser-native method captured before renderer shims', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
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
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
    window.history.replaceState(null, '', '/hermes?view=chat#/session-1')

    expect(installBrowserDesktopBridge()).toBe(true)
    await expect(win.hermesDesktop!.profile.remember(' research ')).resolves.toEqual({ profile: 'research' })

    let url = new URL(window.location.href)
    expect(url.pathname).toBe('/hermes')
    expect(url.searchParams.get('view')).toBe('chat')
    expect(url.searchParams.get('profile')).toBe('research')
    expect(url.hash).toBe('#/session-1')

    await expect(win.hermesDesktop!.profile.remember('default')).resolves.toEqual({ profile: 'default' })
    url = new URL(window.location.href)
    expect(url.searchParams.has('profile')).toBe(false)
    expect(url.hash).toBe('#/session-1')
  })

  it('opens session windows on the owning profile HashRouter route with spectator flags before the hash', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
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

  it.each([
    { expectedProfile: 'work', name: 'absent', opts: undefined },
    { expectedProfile: 'work', name: 'undefined', opts: { profile: undefined } },
    { expectedProfile: 'work', name: 'null', opts: { profile: null } },
    { expectedProfile: 'work', name: 'blank', opts: { profile: '   ' } },
    { expectedProfile: null, name: 'default', opts: { profile: 'default' } },
    { expectedProfile: 'research', name: 'trimmed non-default', opts: { profile: ' research ' } }
  ])('resolves a $name session-window profile against the ambient launch URL', async ({ expectedProfile, opts }) => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
    window.history.replaceState(null, '', '/?profile=work#/existing')
    const open = vi.spyOn(window, 'open').mockImplementation(() => null)

    expect(installBrowserDesktopBridge()).toBe(true)
    await expect(win.hermesDesktop!.openSessionWindow('session-1', opts)).resolves.toEqual({ ok: true })

    const [rawUrl] = open.mock.calls[0]
    expect(new URL(String(rawUrl)).searchParams.get('profile')).toBe(expectedProfile)
  })

  it('exposes one same-origin registry source for Bot profile routing', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
    win.__HERMES_BASE_PATH__ = '/hermes'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(async () =>
        new Response(
          JSON.stringify({
            profiles: [{ name: 'default' }, { name: 'research' }]
          }),
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

  it('maps REST requests to the loopback API with token and profile scoping', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
    win.__HERMES_BASE_PATH__ = '/hermes'

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
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
    win.__HERMES_BASE_PATH__ = '/hermes'
    window.history.replaceState(null, '', '/hermes/?profile=launch-profile#/')

    const worktrees = [{ branch: 'main', detached: false, isMain: true, locked: false, path: '/srv/my repo' }]

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ worktrees }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ diff: 'working diff' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }))

    vi.stubGlobal('fetch', fetchMock)
    expect(installBrowserDesktopBridge()).toBe(true)
    const git = win.hermesDesktop!.git!

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
    await expect(
      git.review.fetchPrComment('/srv/my repo', 'https://github.com/example/repo/pull/1')
    ).resolves.toBeNull()
    expect(fetchMock).toHaveBeenCalledTimes(requestCount)
  })

  it('keeps recovery and filesystem bridge methods safe in browser mode', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
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

  it.each(['token', 'cookie'])('downloads browser files with %s auth on the original profile', async auth => {
    const win = mutableWindow()
    const token = auth === 'token' ? 'served / token' : ''
    win.__HERMES_SESSION_TOKEN__ = token
    win.__HERMES_AUTH_REQUIRED__ = auth === 'cookie'
    win.__HERMES_BASE_PATH__ = '/hermes'
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const downloads: HTMLAnchorElement[] = []
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      downloads.push(this)
    })

    expect(installBrowserDesktopBridge()).toBe(true)
    $connection.set(await win.hermesDesktop!.getConnectionFor!({ connectionId: 'local', profile: 'research' }))
    const pending = downloadGatewayMediaFile('file:///srv/reports/a%20b.pdf')
    $connection.set({ mode: 'remote', profile: 'switched-profile' } as never)
    await expect(pending).resolves.toMatchObject({ saved: true })

    const [requestUrl, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(requestUrl.pathname).toBe('/hermes/api/files/download')
    expect(requestUrl.searchParams.get('path')).toBe('/srv/reports/a b.pdf')
    expect(requestUrl.searchParams.get('profile')).toBe('research')
    expect(init.method).toBe('HEAD')
    expect(init.credentials).toBe('same-origin')
    expect(new Headers(init.headers).get('X-Hermes-Session-Token')).toBe(token || null)
    expect(downloads).toHaveLength(1)
    const url = new URL(downloads[0].href)
    expect(url.origin).toBe(window.location.origin)
    expect(url.pathname).toBe(requestUrl.pathname)
    expect(url.searchParams.get('path')).toBe(requestUrl.searchParams.get('path'))
    expect(url.searchParams.get('profile')).toBe('research')
    expect(url.searchParams.get('token')).toBe(token || null)
    expect(downloads[0].download).toBe('a b.pdf')
    expect(downloads[0].isConnected).toBe(false)

    fetchMock.mockResolvedValue(new Response(null, { status: 404 }))
    await expect(downloadGatewayMediaFile('/srv/missing.pdf')).rejects.toThrow('404')
    await expect(win.hermesDesktop!.saveGatewayFile!({
      connectionId: 'another-host', path: '/srv/reports/a b.pdf'
    })).rejects.toThrow('No connection with id "another-host"')
    expect(downloads).toHaveLength(1)
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it.each(['token', 'cookie'])('streams browser audio/video over HTTP with %s auth', async auth => {
    const win = mutableWindow()
    const token = auth === 'token' ? 'served / token' : ''
    win.__HERMES_SESSION_TOKEN__ = token
    win.__HERMES_AUTH_REQUIRED__ = auth === 'cookie'
    win.__HERMES_BASE_PATH__ = '/hermes'
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)
    $connection.set(await win.hermesDesktop!.getConnectionFor!({ connectionId: 'local', profile: 'research' }))

    for (const path of ['/srv/a b.mp4', '/srv/voice.m4a']) {
      const url = new URL(await resolveMediaPlaybackSrc(`file://${encodeURI(path)}`))
      expect(url.origin).toBe(window.location.origin)
      expect(url.pathname).toBe('/hermes/api/files/stream')
      expect(url.searchParams.get('path')).toBe(path)
      expect(url.searchParams.get('profile')).toBe('research')
      expect(url.searchParams.get('token')).toBe(token || null)
    }

    await expect(resolveMediaPlaybackSrc('https://cdn.example.com/video.mp4')).resolves.toBe(
      'https://cdn.example.com/video.mp4'
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('persists browser image bytes through the existing chat upload API', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
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
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
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
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
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

  it('stages non-image buffers as browser-local object URLs', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
    const createObjectURL = vi.fn((_blob: Blob | MediaSource) => 'blob:http://127.0.0.1:9119/preview')
    vi.spyOn(URL, 'createObjectURL').mockImplementation(createObjectURL)

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
    expect(wrapper).toContain('data:text/html;charset=utf-8;base64,PGgxPnByZXZpZXc8L2gxPg==')
  })

  it('rejects API paths that escape the Webapp origin', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
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
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
    win.__HERMES_BASE_PATH__ = '/hermes'
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
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
    win.__HERMES_BASE_PATH__ = '/hermes'
    $connection.set({ profile: 'active-terminal' } as never)

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 })))

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
    expect(url.pathname).toBe('/hermes/api/pty')
    expect(url.searchParams.get('token')).toBe('served-token')
    expect(url.searchParams.get('profile')).toBe('active-terminal')
    expect(url.searchParams.get('mode')).toBe('shell')
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

    socket.close(1000, 'shell exited')
    expect(exited).toHaveBeenCalledWith({ code: null, signal: null })
    stopData()
    stopExit()
    await expect(win.hermesDesktop!.terminal.dispose(session.id)).resolves.toBe(true)
  })

  it('buffers early host-shell output until Desktop registers its terminal data listener', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 })))

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
    win.__HERMES_SESSION_TOKEN__ = 'served-token'

    expect(installBrowserDesktopBridge()).toBe(true)

    const connection = await win.hermesDesktop!.getConnection('worker-a')
    const ws = new URL(connection.wsUrl)
    expect(connection.mode).toBe('remote')
    expect(connection.profile).toBe('worker-a')
    expect(ws.pathname).toBe('/api/ws')
    expect(ws.searchParams.get('token')).toBe('served-token')
    expect(ws.searchParams.get('profile')).toBe('worker-a')
  })
})
