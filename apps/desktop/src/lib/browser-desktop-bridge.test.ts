import { afterEach, describe, expect, it, vi } from 'vitest'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'

type MutableWindow = Window & {
  __HERMES_BASE_PATH__?: string
  __HERMES_SESSION_TOKEN__?: string
  hermesDesktop?: Window['hermesDesktop']
}

const mutableWindow = () => window as unknown as MutableWindow

afterEach(() => {
  const win = mutableWindow()
  delete win.__HERMES_BASE_PATH__
  delete win.__HERMES_SESSION_TOKEN__
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
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

  it('does not advertise Electron-only capabilities in browser-host mode', () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'

    expect(installBrowserDesktopBridge()).toBe(true)
    expect(win.hermesDesktop?.openSessionInTerminal).toBeUndefined()
    expect(win.hermesDesktop?.connections).toBeUndefined()
    expect(win.hermesDesktop?.getProfileRoutes).toBeUndefined()
    expect(win.hermesDesktop?.onOpenFindBarRequested).toBeUndefined()
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

  it('keeps recovery and filesystem bridge methods safe in browser mode', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'

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
    expect(new Headers(init.headers).get('X-Hermes-Session-Token')).toBe('served-token')
  })

  it('persists browser image bytes through the existing chat upload API', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ path: '/data/data/com.termux/files/home/.hermes/images/upload.png' }), {
        status: 200
      })
    )

    vi.stubGlobal('fetch', fetchMock)

    expect(installBrowserDesktopBridge()).toBe(true)

    const path = await win.hermesDesktop!.saveImageBuffer(
      new Uint8Array([0x89, 0x50, 0x4e, 0x47]),
      '.png'
    )

    expect(path).toBe('/data/data/com.termux/files/home/.hermes/images/upload.png')
    const [requestUrl, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(requestUrl.pathname).toBe('/api/chat/image-upload')
    const payload = JSON.parse(String(init.body)) as { data_url: string; filename: string }
    expect(payload.filename).toBe('desktop-upload.png')
    expect(payload.data_url).toBe('data:image/png;base64,iVBORw==')
  })

  it('stages non-image buffers as browser-local object URLs', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
    const createObjectURL = vi.fn(() => 'blob:http://127.0.0.1:9119/preview')
    vi.spyOn(URL, 'createObjectURL').mockImplementation(createObjectURL)

    expect(installBrowserDesktopBridge()).toBe(true)

    const staged = await win.hermesDesktop!.saveImageBuffer(
      new TextEncoder().encode('<h1>preview</h1>'),
      '.html'
    )

    expect(staged).toBe('blob:http://127.0.0.1:9119/preview')
    expect(createObjectURL).toHaveBeenCalledTimes(1)
  })

  it('maps the browser-hosted terminal rail onto the existing Hermes TUI PTY socket', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'
    win.__HERMES_BASE_PATH__ = '/hermes'

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
    expect(session).toMatchObject({ cwd: '/work', shell: 'hermes-tui' })
    expect(url.pathname).toBe('/hermes/api/pty')
    expect(url.searchParams.get('token')).toBe('served-token')
    expect(url.searchParams.get('cwd')).toBeNull()
    expect(url.searchParams.get('cols')).toBeNull()
    expect(url.searchParams.get('rows')).toBeNull()

    const output = vi.fn()
    const exited = vi.fn()
    const stopData = win.hermesDesktop!.terminal.onData(session.id, output)
    const stopExit = win.hermesDesktop!.terminal.onExit(session.id, exited)
    socket.emitBytes('hello from hermes tui')
    await Promise.resolve()
    expect(output).toHaveBeenCalledWith('hello from hermes tui')

    const multibyte = new TextEncoder().encode('😀')
    socket.emitRaw(multibyte.slice(0, 2))
    socket.emitRaw(multibyte.slice(2))
    expect(output).toHaveBeenCalledWith('😀')
    expect(output).not.toHaveBeenCalledWith(expect.stringContaining('�'))

    await expect(win.hermesDesktop!.terminal.write(session.id, 'hello\r')).resolves.toBe(true)
    await expect(win.hermesDesktop!.terminal.resize(session.id, { cols: 80, rows: 24 })).resolves.toBe(true)
    expect(socket.sent).toEqual(['hello\r', '\u001b[RESIZE:80;24]'])
    await expect(win.hermesDesktop!.terminal.cwd(session.id)).resolves.toBe('/work')

    socket.close(1000, 'tui exited')
    expect(exited).toHaveBeenCalledWith({ code: null, signal: null })
    stopData()
    stopExit()
    await expect(win.hermesDesktop!.terminal.dispose(session.id)).resolves.toBe(true)
  })

  it('buffers early Hermes TUI output until Desktop registers its terminal data listener', async () => {
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
          const bytes = new TextEncoder().encode('Hermes TUI ready')
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
    const output = vi.fn()
    win.hermesDesktop!.terminal.onData(session.id, output)
    await Promise.resolve()
    expect(output).toHaveBeenCalledWith('Hermes TUI ready')
  })

  it('exposes a profile-scoped gateway URL for Desktop boot', async () => {
    const win = mutableWindow()
    win.__HERMES_SESSION_TOKEN__ = 'served-token'

    expect(installBrowserDesktopBridge()).toBe(true)

    const connection = await win.hermesDesktop!.getConnection('worker-a')
    const ws = new URL(connection.wsUrl)
    expect(connection.mode).toBe('local')
    expect(connection.profile).toBe('worker-a')
    expect(ws.pathname).toBe('/api/ws')
    expect(ws.searchParams.get('token')).toBe('served-token')
    expect(ws.searchParams.get('profile')).toBe('worker-a')
  })
})
