import { isGatewayReauthRequired } from '@hermes/shared'
import { afterEach, expect, it, vi } from 'vitest'

import { pluginSocket } from '@/api/plugins'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { consumeWebappSession, watchWebappLaunchLink, WEBAPP_LAUNCH_REQUIRED } from './browser-launch-session'

const win = window as Window & {
  __HERMES_UI_SURFACE__?: string
  __HERMES_BASE_PATH__?: string
  __HERMES_AUTH_REQUIRED__?: boolean
}

afterEach(() => {
  window.dispatchEvent(new Event('beforeunload'))
  Reflect.deleteProperty(window, 'hermesDesktop')
  delete win.__HERMES_UI_SURFACE__
  delete win.__HERMES_BASE_PATH__
  delete win.__HERMES_AUTH_REQUIRED__
  sessionStorage.clear()
  localStorage.clear()
  window.history.replaceState(null, '', '/')
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it('uses one private session for REST, RPC and terminal across reload without losing the profile', async () => {
  const token = 'a'.repeat(43)
  win.__HERMES_UI_SURFACE__ = 'webapp'
  win.__HERMES_BASE_PATH__ = '/one'
  window.history.replaceState(null, '', `/one/?profile=coder#hermes-session=${token}`)
  const urls: URL[] = []

  class Socket {
    onmessage: ((event: { data: string }) => void) | null = null
    constructor(url: URL) {
      urls.push(new URL(url))
      queueMicrotask(() => this.onmessage?.({ data: '\0HERMES_TERMINAL_META:{"terminalId":"shell","pid":1,"cwd":"/work","shell":"sh","reconnected":false}' }))
    }
    close() {}
  }

  vi.stubGlobal('WebSocket', Socket)

  const minted: Record<string, string> = { 'gateway-ticket': 'rpc-ticket', 'terminal-ticket': 'shell-ticket' }

  const fetchMock = vi.fn().mockImplementation(async (url: URL) => {
    const ticket = minted[url.pathname.split('/api/webapp/')[1] ?? '']

    return new Response(ticket ? JSON.stringify({ ticket }) : '{"ok":true}', { status: 200 })
  })

  vi.stubGlobal('fetch', fetchMock)
  expect(installBrowserDesktopBridge()).toBe(true)
  expect(window.location.hash).toBe('')
  expect(window.location.search).toBe('?profile=coder')
  const desktop = window.hermesDesktop!
  await desktop.api({ path: '/api/fs/read-text' })
  expect(new Headers(fetchMock.mock.calls[0][1].headers).get('X-Hermes-Session-Token')).toBe(token)
  expect((await desktop.getConnection()).token).toBe(token)
  // RPC: the gateway socket's one-use ticket is minted over REST with the same private session;
  // the session token itself never rides a socket URL.
  expect(new URL((await desktop.getConnection()).wsUrl).searchParams.has('token')).toBe(false)
  // As the boot dial does (resolveGatewayWsUrl): mint for the descriptor's own profile.
  const rpc = await desktop.getGatewayWsUrl((await desktop.getConnection()).profile)
  const rpcUrl = new URL(typeof rpc === 'string' ? rpc : rpc.ok ? rpc.wsUrl : '')
  expect(rpcUrl.searchParams.get('ticket')).toBe('rpc-ticket')
  expect(rpcUrl.searchParams.get('profile')).toBe('coder')
  const mint = fetchMock.mock.calls.find(([url]) => (url as URL).pathname === '/one/api/webapp/gateway-ticket')!
  expect(new Headers(mint[1].headers).get('X-Hermes-Session-Token')).toBe(token)
  // Terminal: each dial mints its own one-use ticket the same way.
  await desktop.terminal.start({ cwd: '/work' })
  expect(urls[0].pathname).toBe('/one/api/host-terminal')
  expect(urls[0].searchParams.get('ticket')).toBe('shell-ticket')
  const shellMint = fetchMock.mock.calls.find(([url]) => (url as URL).pathname === '/one/api/webapp/terminal-ticket')!
  expect(shellMint[1].method).toBe('POST')
  expect(new Headers(shellMint[1].headers).get('X-Hermes-Session-Token')).toBe(token)
  // A plugin's live socket cannot carry a ticket here; it stays on its polling fallback.
  const stopPluginSocket = pluginSocket('kanban', '/events', () => undefined)
  await new Promise(resolve => setTimeout(resolve, 20))
  stopPluginSocket()
  // No URL this tab requested or dialed carries the session token: it rides only the request header.
  const requested = [...fetchMock.mock.calls.map(([url]) => String(url)), ...urls.map(String), rpcUrl.href]
  expect(requested.filter(url => url.includes(token))).toEqual([])
  expect(urls).toHaveLength(1)
  window.dispatchEvent(new Event('beforeunload'))
  Reflect.deleteProperty(window, 'hermesDesktop')
  window.history.replaceState(null, '', '/one/?profile=coder#/settings')
  expect(installBrowserDesktopBridge()).toBe(true)
  expect((await window.hermesDesktop!.getConnection()).token).toBe(token)
  expect(window.location.hash).toBe('#/settings')
  expect(consumeWebappSession('/other')).toBe('')
  expect(Object.values(localStorage)).not.toContain(token)
})

it('fails closed with actionable launch instructions in a fresh tab and ignores local grants under OAuth', async () => {
  win.__HERMES_UI_SURFACE__ = 'webapp'
  const fetchMock = vi.fn()
  vi.stubGlobal('fetch', fetchMock)
  expect(installBrowserDesktopBridge()).toBe(true)
  await expect(window.hermesDesktop!.getConnection()).rejects.toThrow('Run hermes webapp')
  await expect(window.hermesDesktop!.api({ path: '/api/fs/read-text' })).rejects.toThrow('Run hermes webapp')
  expect(fetchMock).not.toHaveBeenCalled()
  window.dispatchEvent(new Event('beforeunload'))
  Reflect.deleteProperty(window, 'hermesDesktop')
  window.history.replaceState(null, '', `/#hermes-session=${'b'.repeat(43)}`)
  expect(consumeWebappSession('')).toHaveLength(43)
  window.history.replaceState(null, '', '/#hermes-session=invalid')
  expect(consumeWebappSession('')).toBe('')
  expect(consumeWebappSession('')).toBe('')
  win.__HERMES_AUTH_REQUIRED__ = true
  expect(installBrowserDesktopBridge()).toBe(true)
  expect((await window.hermesDesktop!.getConnection()).authMode).toBe('oauth')
  expect((await window.hermesDesktop!.getConnection()).token).toBe('')
})

it('reloads for a pasted launch fragment but not HashRouter navigation', () => {
  const reload = vi.fn()
  watchWebappLaunchLink(reload)
  window.history.replaceState(null, '', '/#/settings')
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  expect(reload).not.toHaveBeenCalled()
  const fragment = `#hermes-session=${'c'.repeat(43)}`
  window.history.replaceState(null, '', `/${fragment}`)
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  expect(reload).toHaveBeenCalledOnce()
  expect(window.location.hash).toBe(fragment)
  window.dispatchEvent(new Event('beforeunload'))
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  expect(reload).toHaveBeenCalledOnce()
})

it.each(['window', 'session'] as const)('hands off a private %s through a one-use ticket without sharing its session token', async kind => {
  const token = 'a'.repeat(43)
  const ticket = 'b'.repeat(43)
  win.__HERMES_UI_SURFACE__ = 'webapp'
  win.__HERMES_BASE_PATH__ = '/one'
  window.history.replaceState(null, '', `/one/?profile=coder#hermes-session=${token}`)
  const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ticket })))
  vi.stubGlobal('fetch', fetchMock)

  class Channel {
    static current: Channel
    onmessage: ((event: { data: unknown }) => void) | null = null
    constructor(readonly name: string) { Channel.current = this }
    postMessage = vi.fn(() => queueMicrotask(() => this.onmessage?.({ data: { type: 'done' } })))
    close = vi.fn()
  }
  vi.stubGlobal('BroadcastChannel', Channel)
  const open = vi.spyOn(window, 'open').mockReturnValue(null)
  expect(installBrowserDesktopBridge()).toBe(true)

  const pending = kind === 'window'
    ? window.hermesDesktop.openWindow({ connectionId: 'local', profile: 'research' })
    : window.hermesDesktop.openSessionWindow('session / 1', { profile: 'research', watch: true })

  // The window opens before any await, preserving the browser user gesture.
  expect(open).toHaveBeenCalledOnce()
  expect(fetchMock).not.toHaveBeenCalled()
  Channel.current.onmessage?.({ data: { type: 'ready' } })
  await pending

  expect(fetchMock).toHaveBeenCalledOnce()
  const [url, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
  expect(url.pathname).toBe('/one/api/webapp/window-ticket')
  expect(init.method).toBe('POST')
  expect(new Headers(init.headers).get('X-Hermes-Session-Token')).toBe(token)
  expect(open).toHaveBeenCalledOnce()
  const [opened, target, features] = open.mock.calls[0]
  const child = new URL(String(opened))
  expect(child.origin).toBe(window.location.origin)
  expect(child.pathname).toBe('/one/webapp/window')
  expect(child.search).toBe('')
  const handoff = new URLSearchParams(child.hash.slice(1))
  expect(Channel.current.name).toBe(`hermes.webapp.window:${handoff.get('channel')}`)
  expect(Channel.current.postMessage).toHaveBeenCalledWith({ ticket })
  expect(Channel.current.close).toHaveBeenCalledOnce()
  expect(handoff.get('query')).toContain('profile=research')
  expect(handoff.get('route')).toBe(kind === 'window' ? '/' : '/session%20%2F%201')
  expect(handoff.get('query')?.includes('watch=1')).toBe(kind === 'session')
  expect([target, features]).toEqual(['_blank', 'noopener,noreferrer'])
  expect(String(opened)).not.toContain(token)
  expect(String(opened)).not.toContain(ticket)
  expect(window.location.search).toBe('?profile=coder')
  expect(Object.values(localStorage)).not.toContain(token)

  fetchMock.mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }))
  open.mockClear()
  const failed = window.hermesDesktop.openWindow()
  Channel.current.onmessage?.({ data: { type: 'ready' } })
  await expect(failed).rejects.toThrow()
  expect(Channel.current.postMessage).toHaveBeenCalledWith({ error: expect.any(String) })
  expect(Channel.current.close).toHaveBeenCalledOnce()
})

it('asks a tab without a working credential to sign in, so reconnecting stops', async () => {
  // The reconnect loop stops retrying only for an error that reads as a sign-in requirement;
  // the startup overlay then shows the Webapp sign-in view for this message.
  const failure = (request: Promise<unknown>) => request.then(() => null, (error: unknown) => error)
  win.__HERMES_UI_SURFACE__ = 'webapp'
  vi.stubGlobal('fetch', vi.fn(async () => new Response('Unauthorized', { status: 401 })))
  expect(installBrowserDesktopBridge()).toBe(true)

  for (const error of [
    await failure(window.hermesDesktop!.getConnection()),
    await failure(window.hermesDesktop!.api({ path: '/api/status' }))
  ]) {
    expect(isGatewayReauthRequired(error)).toBe(true)
    expect((error as Error).message).toBe(WEBAPP_LAUNCH_REQUIRED)
  }

  // hermes webapp --sign-out replaced the credential this tab holds.
  window.dispatchEvent(new Event('beforeunload'))
  Reflect.deleteProperty(window, 'hermesDesktop')
  window.history.replaceState(null, '', `/#hermes-session=${'d'.repeat(43)}`)
  expect(installBrowserDesktopBridge()).toBe(true)
  const signedOut = await failure(window.hermesDesktop!.api({ path: '/api/status' }))

  expect(isGatewayReauthRequired(signedOut)).toBe(true)
  expect((signedOut as Error).message).toBe(WEBAPP_LAUNCH_REQUIRED)
})
