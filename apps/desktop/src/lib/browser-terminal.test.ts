import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'

const META = '\u0000HERMES_TERMINAL_META:'

class Socket {
  static OPEN = 1
  static instances: Socket[] = []
  static reply = (socket: Socket) => socket.metadata()
  readyState = 1
  binaryType = ''
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: (() => void) | null = null
  sent: string[] = []
  url: URL

  constructor(url: string | URL) {
    this.url = new URL(url)
    Socket.instances.push(this)
    queueMicrotask(() => Socket.reply(this))
  }

  metadata(extra: Record<string, unknown> = {}) {
    this.onmessage?.({ data: META + JSON.stringify({
      shell: 'bash', cwd: '/original', terminalId: 'opaque-server-id', retentionSeconds: 900,
      reconnected: this.url.searchParams.has('attach'), ...extra
    }) } as MessageEvent)
  }

  output(text: string) {
    this.onmessage?.({ data: new TextEncoder().encode(text).buffer } as MessageEvent)
  }

  close(code = 1000, reason = '') {
    this.readyState = 3
    this.onclose?.(new CloseEvent('close', { code, reason }))
  }

  send(data: string) { this.sent.push(data) }
}

function install() {
  Reflect.deleteProperty(window, 'hermesDesktop')
  Object.assign(window, { __HERMES_AUTH_REQUIRED__: true, __HERMES_BASE_PATH__: '/one' })
  installBrowserDesktopBridge()

  return window.hermesDesktop!.terminal
}

beforeEach(() => {
  localStorage.clear()
  vi.useFakeTimers()
  Socket.instances = []
  Socket.reply = socket => socket.metadata()
  vi.stubGlobal('WebSocket', Socket)
  let tickets = 0
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ ticket: `fresh-${++tickets}` }))))
  $connection.set({ profile: 'original' } as never)
})

afterEach(() => {
  window.dispatchEvent(new Event('beforeunload'))
  Reflect.deleteProperty(window, 'hermesDesktop')
  Reflect.deleteProperty(window, '__HERMES_AUTH_REQUIRED__')
  Reflect.deleteProperty(window, '__HERMES_BASE_PATH__')
  $connection.set(null)
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  localStorage.clear()
})

describe('persistent browser terminals', () => {
  it('consumes an empty replay frame without suppressing the first live output', async () => {
    const api = install()
    const session = await api.start({ cwd: '/original', restoreKey: 'empty-history' })
    const output = vi.fn()
    api.onData(session.id, output)
    Socket.instances[0].output('')
    Socket.instances[0].output('\u001b[6n')
    expect(output).toHaveBeenCalledExactlyOnceWith('\u001b[6n')
  })

  it('reattaches the saved tab after reload to its captured profile, not the newly active one', async () => {
    const firstApi = install()
    await firstApi.start({ cwd: '/original', restoreKey: 'tab-one' })
    window.dispatchEvent(new Event('beforeunload'))
    $connection.set({ profile: 'other' } as never)
    const secondApi = install()
    const session = await secondApi.start({ cwd: '/other', restoreKey: 'tab-one' })
    const socket = Socket.instances.at(-1)!
    expect(socket.url.searchParams.get('attach')).toBe('opaque-server-id')
    expect(socket.url.searchParams.get('profile')).toBe('original')
    expect(session.cwd).toBe('/original')
    expect(socket.url.searchParams.get('ticket')).not.toBe(Socket.instances[0].url.searchParams.get('ticket'))
    expect(Socket.instances.some(s => s.url.searchParams.get('action') === 'close')).toBe(false)
  })

  it('automatically reconnects without reporting exit, resets before replay, and does not send input offline', async () => {
    const api = install()
    const session = await api.start({ cwd: '/original', restoreKey: 'tab-one' })
    const output = vi.fn()
    const exit = vi.fn()
    api.onData(session.id, output)
    api.onExit(session.id, exit)
    Socket.instances[0].output('before')
    Socket.instances[0].close(1006)
    await expect(api.write(session.id, 'dangerous command\r')).resolves.toBe(false)
    await api.resize(session.id, { cols: 97, rows: 31 })
    await vi.advanceTimersByTimeAsync(1000)
    expect(Socket.instances).toHaveLength(2)
    const resumed = Socket.instances[1]
    expect(resumed.url.searchParams.get('attach')).toBe('opaque-server-id')
    expect(resumed.url.searchParams.get('cols')).toBe('97')
    expect(resumed.sent).toContain('\u001b[RESIZE:97;31]')
    resumed.output('before\r\nafter')
    const frames = output.mock.calls.map(([data]) => data as string)
    expect(frames.at(-2)).toContain('\u001bc')
    expect(frames.at(-1)).toBe('before\r\nafter')
    expect(exit).not.toHaveBeenCalled()
    expect(resumed.sent.join('')).not.toContain('dangerous command')
    expect(resumed.sent.join('')).not.toContain('\u000c')
  })

  it('closes explicitly with a fresh authenticated request even while disconnected', async () => {
    const api = install()
    const session = await api.start({ cwd: '/original', restoreKey: 'tab-one' })
    Socket.instances[0].close(1006)

    Socket.reply = socket => {
      socket.metadata({ closed: true })
      socket.close()
    }

    await expect(api.dispose(session.id)).resolves.toBe(true)
    const close = Socket.instances.at(-1)!
    expect(close.url.searchParams.get('action')).toBe('close')
    expect(close.url.searchParams.get('attach')).toBe('opaque-server-id')
    expect(close.url.searchParams.get('profile')).toBe('original')
    await vi.advanceTimersByTimeAsync(30_000)
    expect(Socket.instances).toHaveLength(2)
    expect(localStorage.length).toBe(0)
  })

  it('stops on supersession rather than fighting the other tab, and retains the resume identity', async () => {
    const api = install()
    const session = await api.start({ cwd: '/original', restoreKey: 'tab-one' })
    const exit = vi.fn()
    api.onExit(session.id, exit)
    Socket.instances[0].close(4409)
    await vi.advanceTimersByTimeAsync(30_000)
    expect(Socket.instances).toHaveLength(1)
    expect(exit).toHaveBeenCalledWith({ code: null, signal: 'superseded' })
    expect(localStorage.length).toBe(1)
  })

  it('reports expired resumes without silently creating a replacement shell', async () => {
    const api = install()
    const session = await api.start({ cwd: '/original', restoreKey: 'tab-one' })
    const exit = vi.fn()
    api.onExit(session.id, exit)
    Socket.instances[0].close(1006)
    Socket.reply = socket => socket.close(4410, 'Terminal expired')
    await vi.advanceTimersByTimeAsync(30_000)
    expect(Socket.instances).toHaveLength(2)
    expect(exit).toHaveBeenCalledWith({ code: null, signal: 'expired' })
    expect(localStorage.length).toBe(0)
  })

  it('never retries denied attachments or silently adopts the active profile', async () => {
    const api = install()
    const session = await api.start({ cwd: '/original', restoreKey: 'tab-one' })
    Socket.instances[0].close(1006)
    const exit = vi.fn()
    api.onExit(session.id, exit)
    $connection.set({ profile: 'other' } as never)
    Socket.reply = socket => socket.close(4403)
    await vi.advanceTimersByTimeAsync(30_000)
    expect(Socket.instances).toHaveLength(2)
    expect(exit).toHaveBeenCalledWith({ code: null, signal: 'denied' })
    expect(Socket.instances[1].url.searchParams.get('profile')).toBe('original')
  })

  it('bounds network retries and resumes after bfcache without duplicating viewers', async () => {
    const api = install()
    const session = await api.start({ cwd: '/original', restoreKey: 'tab-one' })
    const state = vi.fn()
    const exit = vi.fn()
    api.onState!(session.id, state)
    api.onExit(session.id, exit)
    window.dispatchEvent(new Event('beforeunload'))
    window.dispatchEvent(new PageTransitionEvent('pagehide', { persisted: true }))
    await vi.advanceTimersByTimeAsync(30_000)
    expect(Socket.instances).toHaveLength(1)
    window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true }))
    await vi.advanceTimersByTimeAsync(1000)
    expect(Socket.instances).toHaveLength(2)
    expect(Socket.instances[1].url.searchParams.get('attach')).toBe('opaque-server-id')
    Socket.reply = socket => socket.close(1006)
    Socket.instances[1].close(1006)
    await vi.advanceTimersByTimeAsync(120_000)
    expect(state).toHaveBeenCalledWith('reconnecting')
    expect(state).toHaveBeenLastCalledWith('disconnected')
    expect(exit).toHaveBeenCalledWith({ code: null, signal: 'disconnected' })
    const attempts = Socket.instances.length
    expect(attempts).toBeLessThanOrEqual(7)
    await vi.advanceTimersByTimeAsync(120_000)
    expect(Socket.instances).toHaveLength(attempts)
    expect(Socket.instances.slice(1).every(socket => socket.url.searchParams.has('attach'))).toBe(true)
  })

  it('waits for close acknowledgement and preserves the handle on a denied close', async () => {
    const api = install()
    await api.start({ cwd: '/original', restoreKey: 'tab-one' })
    Socket.reply = socket => socket.close(4403)
    await expect(api.closeSaved!('tab-one')).rejects.toThrow(/close/i)
    expect(localStorage.length).toBe(1)

    Socket.reply = socket => { socket.metadata({ closed: true }); socket.close() }
    await expect(api.closeSaved!('tab-one')).resolves.toBe(true)
    expect(localStorage.length).toBe(0)
  })

  it('does not redial a late auth ticket after detach', async () => {
    const api = install()
    const session = await api.start({ cwd: '/original', restoreKey: 'tab-one' })
    let finishTicket!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(resolve => { finishTicket = resolve })))
    Socket.instances[0].close(1006)
    await vi.advanceTimersByTimeAsync(1000)
    await api.detach!(session.id)
    finishTicket(new Response(JSON.stringify({ ticket: 'late' })))
    await vi.advanceTimersByTimeAsync(30_000)
    expect(Socket.instances).toHaveLength(1)
  })

  it('does not share persisted terminals between installations behind different proxy prefixes', async () => {
    await install().start({ cwd: '/original', restoreKey: 'tab-one' })
    window.dispatchEvent(new Event('beforeunload'))
    Reflect.deleteProperty(window, 'hermesDesktop')
    Object.assign(window, { __HERMES_BASE_PATH__: '/two' })
    installBrowserDesktopBridge()
    await window.hermesDesktop!.terminal.start({ cwd: '/original', restoreKey: 'tab-one' })
    expect(Socket.instances.at(-1)!.url.searchParams.has('attach')).toBe(false)
  })
})
