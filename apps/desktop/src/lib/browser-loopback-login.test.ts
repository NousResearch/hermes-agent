import { afterEach, describe, expect, it, vi } from 'vitest'

import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'
import { WEBAPP_LAUNCH_REQUIRED } from '@/lib/browser-launch-session'
import { browserBootstrap, BrowserReauthRequiredError } from '@/lib/browser-transport'
import { bootRenderer } from '@/lib/ui-preferences/boot'
import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

type LoopbackWindow = Window & {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_BASE_PATH__?: string
  __HERMES_LOGIN_AVAILABLE__?: boolean
  __HERMES_UI_SURFACE__?: string
  hermesDesktop?: Window['hermesDesktop']
}

const loopbackWindow = () => window as unknown as LoopbackWindow

afterEach(() => {
  const win = loopbackWindow()
  delete win.__HERMES_AUTH_REQUIRED__
  delete win.__HERMES_BASE_PATH__
  delete win.__HERMES_LOGIN_AVAILABLE__
  clearWebappSession()
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  $connection.set(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('loopback Webapp login recovery and launch-link coexistence', () => {
  it.each([
    { basePath: '', expectedApiPath: '/api/profiles', expectedLoginPath: '/login', loginUrl: '/login' },
    {
      basePath: '/hermes',
      expectedApiPath: '/hermes/api/profiles',
      expectedLoginPath: '/hermes/login',
      loginUrl: '/hermes/login'
    }
  ])(
    'recovers to /login after cookie expiry reload when interactive login is available (basePath="$basePath")',
    async ({ basePath, expectedApiPath, expectedLoginPath, loginUrl }) => {
      const win = loopbackWindow()
      win.__HERMES_UI_SURFACE__ = 'webapp'
      win.__HERMES_AUTH_REQUIRED__ = false
      win.__HERMES_LOGIN_AVAILABLE__ = true

      if (basePath) {
        win.__HERMES_BASE_PATH__ = basePath
      }

      const reauthUrls: string[] = []

      const onReauth = (event: Event) => {
        event.preventDefault()
        reauthUrls.push(String((event as CustomEvent<{ loginUrl?: string }>).detail.loginUrl || ''))
      }

      window.addEventListener('hermes:browser-reauth-required', onReauth)

      const fetchMock = vi.fn().mockImplementation(async () =>
        new Response(
          JSON.stringify({
            detail: 'Unauthorized',
            error: 'unauthenticated',
            login_url: loginUrl,
            reason: 'no_cookie'
          }),
          {
            headers: { 'content-type': 'application/json' },
            status: 401
          }
        )
      )

      vi.stubGlobal('fetch', fetchMock)

      try {
        expect(browserBootstrap()).toEqual({
          authRequired: true,
          basePath,
          privateSession: false,
          token: ''
        })
        expect(installBrowserDesktopBridge()).toBe(true)

        // connectionFor() must NOT throw WEBAPP_LAUNCH_REQUIRED before the browser can reach the server.
        const connection = await win.hermesDesktop!.getConnection()
        expect(connection.authMode).toBe('oauth')
        expect(connection.token).toBe('')

        // browserFetch() sends the request and navigates to login_url on the structured 401 envelope.
        await expect(win.hermesDesktop!.api({ path: '/api/profiles' })).rejects.toBeInstanceOf(
          BrowserReauthRequiredError
        )
        const [profileUrl, profileInit] = fetchMock.mock.calls[0] as [URL, RequestInit]
        expect(profileUrl.pathname).toBe(expectedApiPath)
        expect(profileInit.credentials).toBe('same-origin')
        expect(new Headers(profileInit.headers).has('X-Hermes-Session-Token')).toBe(false)

        // Gateway socket-ticket mint returns the actionable OAuth reauth signal instead of launch-link error.
        await expect(win.hermesDesktop!.getGatewayWsUrl('worker-a')).resolves.toMatchObject({
          error: expect.stringContaining('unauthenticated'),
          needsOauthLogin: true,
          ok: false
        })

        expect(reauthUrls).toHaveLength(2)
        expect(reauthUrls.every(href => new URL(href).pathname === expectedLoginPath)).toBe(true)
      } finally {
        window.removeEventListener('hermes:browser-reauth-required', onReauth)
      }
    }
  )

  it('triggers login recovery during early bootRenderer() UI preferences fetch after cookie expiry', async () => {
    const win = loopbackWindow()
    win.__HERMES_UI_SURFACE__ = 'webapp'
    win.__HERMES_AUTH_REQUIRED__ = false
    win.__HERMES_LOGIN_AVAILABLE__ = true

    const reauthUrls: string[] = []

    const onReauth = (event: Event) => {
      event.preventDefault()
      reauthUrls.push(String((event as CustomEvent<{ loginUrl?: string }>).detail.loginUrl || ''))
    }

    window.addEventListener('hermes:browser-reauth-required', onReauth)

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: 'Unauthorized',
          error: 'unauthenticated',
          login_url: '/login',
          reason: 'no_cookie'
        }),
        { status: 401 }
      )
    )

    vi.stubGlobal('fetch', fetchMock)

    try {
      await bootRenderer(async () => 'loaded')
      expect(fetchMock).toHaveBeenCalledTimes(1)
      expect((fetchMock.mock.calls[0] as [URL])[0].pathname).toBe('/api/ui-preferences')
      expect(reauthUrls).toHaveLength(1)
      expect(new URL(reauthUrls[0]!).pathname).toBe('/login')
    } finally {
      window.removeEventListener('hermes:browser-reauth-required', onReauth)
    }
  })

  it('keeps launch-link token sessions working when interactive login is also configured', async () => {
    const win = loopbackWindow()
    win.__HERMES_AUTH_REQUIRED__ = false
    win.__HERMES_LOGIN_AVAILABLE__ = true
    seedWebappSession('launch-token')

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ profiles: [{ name: 'default' }] }), { status: 200 }))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ detail: 'Unauthorized', error: 'unauthenticated', login_url: '/login', reason: 'no_cookie' }),
          { status: 401 }
        )
      )

    vi.stubGlobal('fetch', fetchMock)

    expect(browserBootstrap()).toEqual({
      authRequired: false,
      basePath: '',
      privateSession: true,
      token: 'launch-token'
    })
    expect(installBrowserDesktopBridge()).toBe(true)

    const connection = await win.hermesDesktop!.getConnection()
    expect(connection.authMode).toBe('token')
    expect(connection.token).toBe('launch-token')

    await expect(win.hermesDesktop!.api({ path: '/api/profiles' })).resolves.toEqual({
      profiles: [{ name: 'default' }]
    })
    const [, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(new Headers(init.headers).get('X-Hermes-Session-Token')).toBe('launch-token')

    // A revoked launch-link token still shows launch-link sign-in instructions.
    await expect(win.hermesDesktop!.api({ path: '/api/profiles' })).rejects.toThrow(WEBAPP_LAUNCH_REQUIRED)
  })

  it('prefers verified cookie auth over a stale launch-link token left in sessionStorage', async () => {
    const win = loopbackWindow()
    win.__HERMES_AUTH_REQUIRED__ = true
    win.__HERMES_LOGIN_AVAILABLE__ = true
    seedWebappSession('stale-launch-token')

    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    expect(browserBootstrap()).toEqual({
      authRequired: true,
      basePath: '',
      privateSession: false,
      token: ''
    })
    expect(installBrowserDesktopBridge()).toBe(true)

    await win.hermesDesktop!.api({ path: '/api/profiles' })
    const [, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(init.credentials).toBe('same-origin')
    expect(new Headers(init.headers).has('X-Hermes-Session-Token')).toBe(false)
  })

  it('keeps launch-link-only behavior when no interactive login provider is configured', async () => {
    const win = loopbackWindow()
    win.__HERMES_UI_SURFACE__ = 'webapp'
    win.__HERMES_AUTH_REQUIRED__ = false

    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    expect(browserBootstrap()).toEqual({
      authRequired: false,
      basePath: '',
      privateSession: true,
      token: ''
    })
    expect(installBrowserDesktopBridge()).toBe(true)

    await expect(win.hermesDesktop!.getConnection()).rejects.toThrow(WEBAPP_LAUNCH_REQUIRED)
    await expect(win.hermesDesktop!.api({ path: '/api/profiles' })).rejects.toThrow(WEBAPP_LAUNCH_REQUIRED)
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
