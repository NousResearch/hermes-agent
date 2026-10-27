import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { resolveSiblingWsUrl } from './sibling-ws-url'
import { resolveSpeakStreamUrl } from './voice-playback'

// The browser-hosted Desktop opens each voice stream with a one-use ticket minted for that stream.
// A socket ticket opens only the socket it names, so the gateway's ticket (named for `/api/ws`)
// never opens `/api/audio/speak-stream` or `/api/audio/transcribe-stream`; dialing them with it
// sent streaming voice to its whole-reply fallback every time.

type MutableWindow = Window & {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_BASE_PATH__?: string
  hermesDesktop?: Window['hermesDesktop']
}

const mutableWindow = () => window as unknown as MutableWindow

function serveTickets() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    const url = new URL(String(input))

    const ticket = url.pathname.endsWith('/api/webapp/socket-ticket') ? `${url.searchParams.get('socket')}-ticket` : ''

    return new Response(JSON.stringify(ticket ? { ticket, ttl_seconds: 60 } : {}), {
      headers: { 'content-type': 'application/json' },
      status: ticket ? 200 : 404
    })
  })

  vi.stubGlobal('fetch', fetchMock)

  return fetchMock
}

const minted = (fetchMock: ReturnType<typeof serveTickets>) =>
  fetchMock.mock.calls.map(([input, init]) => ({ init, url: new URL(String(input)) }))

afterEach(() => {
  const win = mutableWindow()
  delete win.__HERMES_AUTH_REQUIRED__
  delete win.__HERMES_BASE_PATH__
  clearWebappSession()
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('browser-hosted Desktop voice streams', () => {
  it('speaks on a ticket minted for the speech stream, never the gateway ticket', async () => {
    const win = mutableWindow()
    win.__HERMES_AUTH_REQUIRED__ = true
    win.__HERMES_BASE_PATH__ = '/hermes'
    const fetchMock = serveTickets()
    expect(installBrowserDesktopBridge()).toBe(true)
    setApiRequestProfile('worker-a')

    const url = new URL((await resolveSpeakStreamUrl())!)

    expect(url.pathname).toBe('/hermes/api/audio/speak-stream')
    expect(url.searchParams.get('ticket')).toBe('speak-stream-ticket')
    expect(url.searchParams.get('profile')).toBe('worker-a')
    expect(url.searchParams.has('token')).toBe(false)

    const [mint, ...others] = minted(fetchMock)
    expect(others).toEqual([])
    expect(mint.url.pathname).toBe('/hermes/api/webapp/socket-ticket')
    expect(mint.url.searchParams.get('socket')).toBe('speak-stream')
    expect(mint.init?.method).toBe('POST')
    expect(mint.init?.credentials).toBe('same-origin')
  })

  it('dictates on a ticket minted for the dictation stream, the session token kept out of the URL', async () => {
    seedWebappSession('served-token')
    const fetchMock = serveTickets()
    expect(installBrowserDesktopBridge()).toBe(true)

    const url = new URL(await resolveSiblingWsUrl({ profile: 'worker-a' }, '/api/audio/transcribe-stream'))

    expect(url.pathname).toBe('/api/audio/transcribe-stream')
    expect(url.searchParams.get('ticket')).toBe('transcribe-stream-ticket')
    expect(url.searchParams.get('profile')).toBe('worker-a')
    expect(url.toString()).not.toContain('served-token')

    const [mint, ...others] = minted(fetchMock)
    expect(others).toEqual([])
    expect(mint.url.pathname).toBe('/api/webapp/socket-ticket')
    expect(mint.url.searchParams.get('socket')).toBe('transcribe-stream')
    expect(new Headers(mint.init?.headers).get('X-Hermes-Session-Token')).toBe('served-token')
  })

  it('leaves the gateway socket on its own ticket', async () => {
    const win = mutableWindow()
    win.__HERMES_AUTH_REQUIRED__ = true
    const fetchMock = serveTickets()
    expect(installBrowserDesktopBridge()).toBe(true)

    const result = await win.hermesDesktop!.getGatewayWsUrl()
    const url = new URL(typeof result === 'string' ? result : result.ok ? result.wsUrl : '')

    expect(url.pathname).toBe('/api/ws')
    expect(url.searchParams.get('ticket')).toBe('gateway-ticket')
    expect(minted(fetchMock).map(({ url }) => [url.pathname, url.searchParams.get('socket')])).toEqual([
      ['/api/webapp/socket-ticket', 'gateway']
    ])
  })
})
