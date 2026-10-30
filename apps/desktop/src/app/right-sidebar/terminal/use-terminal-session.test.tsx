import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'
import { WEBAPP_LAUNCH_REQUIRED } from '@/lib/browser-launch-session'
import { $desktopBoot } from '@/store/boot'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import {
  $terminals,
  closeAllTerminals,
  closeOtherTerminals,
  createTerminal,
  updateTerminalReviveBuffer
} from './terminals'
import { useTerminalSession } from './use-terminal-session'

const emulator = vi.hoisted(() => ({
  input: (_data: string) => {},
  output: '',
  latestFontFamilyRef: { current: 'monospace' },
  mountedRef: { current: false }
}))

vi.mock('@xterm/xterm', () => ({
  Terminal: class {
    options: Record<string, unknown>
    cols = 80
    rows = 24
    unicode = { activeVersion: '11' }
    modes = { mouseTrackingMode: 'none' }
    buffer = { active: { type: 'normal' } }
    parser = { registerOscHandler: () => ({ dispose() {} }) }
    constructor(options: Record<string, unknown>) {
      this.options = options
    }
    loadAddon() {}
    open() {}
    focus() {}
    dispose() {}
    refresh() {}
    clearSelection() {}
    getSelection() {
      return ''
    }
    hasSelection() {
      return false
    }
    registerMarker() {
      return { line: 0, dispose() {} }
    }
    onKey() {
      return { dispose() {} }
    }
    onData(callback: (data: string) => void) {
      emulator.input = callback

      return { dispose() {} }
    }
    onSelectionChange() {
      return { dispose() {} }
    }
    attachCustomKeyEventHandler() {}
    write(data: string, callback?: () => void) {
      if (data === '\u001bc') {
        emulator.output = ''
      }

      if (data.includes('\u001b[6n')) {
        emulator.input('\u001b[1;1R')
      }
      emulator.output += data
      callback?.()
    }
  }
}))
vi.mock('@xterm/addon-fit', () => ({
  FitAddon: class {
    fit() {}
  }
}))
vi.mock('@xterm/addon-serialize', () => ({
  SerializeAddon: class {
    serialize() {
      return '$ make\r\nbuilt ok\r\n'
    }
  }
}))
vi.mock('@xterm/addon-unicode11', () => ({ Unicode11Addon: class {} }))
vi.mock('@xterm/addon-webgl', () => ({
  WebglAddon: class {
    onContextLoss() {}
    clearTextureAtlas() {}
  }
}))
vi.mock('./links', () => ({ terminalLinkHandler: {}, terminalWebLinksAddon: () => ({}) }))
vi.mock('./buffer', () => ({ makeTerminalReader: () => () => '', registerTerminalReader: () => () => {} }))
vi.mock('./terminal-context-menu', () => ({ registerTerminalContextMenu: () => () => {} }))
vi.mock('./terminal-font', () => ({ prepareTerminalFontFamily: async () => 'monospace' }))
vi.mock('./use-terminal-font', () => ({ useTerminalFontController: () => emulator }))
vi.mock('@/themes/context', () => ({ useTheme: () => ({ renderedMode: 'dark', theme: {}, themeName: 'test' }) }))

class Socket {
  static OPEN = 1
  static instances: Socket[] = []
  static persistent = false
  // Set: every new socket is closed with this instead of answering.
  static refusal: [number, string] | null = null
  url: URL
  sent: string[] = []
  readyState = 1
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  constructor(url: string | URL) {
    this.url = new URL(url)
    Socket.instances.push(this)
    queueMicrotask(() => {
      if (Socket.refusal) {
        this.close(...Socket.refusal)

        return
      }

      const metadata = Socket.persistent
        ? {
            shell: 'fish',
            cwd: '/work',
            terminalId: this.url.searchParams.get('attach') || `shell-${Socket.instances.length}`,
            closed: this.url.searchParams.get('action') === 'close'
          }
        : { shell: 'fish' }

      this.onmessage?.({ data: '\u0000HERMES_TERMINAL_META:' + JSON.stringify(metadata) } as MessageEvent)

      if ('closed' in metadata && metadata.closed) {
        this.close()
      }
    })
  }
  send(data: string) {
    this.sent.push(data)
  }
  close(code = 1000, reason = '') {
    this.readyState = 3
    this.onclose?.(new CloseEvent('close', { code, reason }))
  }
  output(data: string) {
    this.onmessage?.({ data } as MessageEvent)
  }
}

// Real scrollback (a command and its output): main's idle-prompt heuristic would clear a lone line.
const SAVED_HISTORY = '$ echo saved history\r\nsaved history'

function Harness({ id }: { id: string }) {
  const { hostRef, status } = useTerminalSession({
    id,
    cwd: '/work',
    active: false,
    reviveBuffer: SAVED_HISTORY,
    onAddSelectionToChat: () => {}
  })

  return (
    <>
      <span>{status}</span>
      <div ref={hostRef} />
    </>
  )
}

beforeEach(() => {
  window.dispatchEvent(new PageTransitionEvent('pageshow'))
  localStorage.clear()
  // A Webapp tab dials each host terminal with a one-use ticket from the ticket route.
  const realFetch = globalThis.fetch
  let minted = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const mint = new URL(String(input), window.location.href)

      if (mint.pathname.endsWith('/api/webapp/socket-ticket') && mint.searchParams.get('socket') === 'host-terminal') {
        minted += 1

        return new Response(JSON.stringify({ ticket: `shell-ticket-${minted}` }), { status: 200 })
      }

      return realFetch(input, init)
    })
  )
})

afterEach(() => {
  cleanup()
  window.dispatchEvent(new Event('beforeunload'))
  Reflect.deleteProperty(window, 'hermesDesktop')
  clearWebappSession()
  $terminals.set([])
  Socket.instances = []
  Socket.persistent = false
  Socket.refusal = null
  emulator.output = ''
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it.each([1006, 1000])(
  'retains the saved tab on transport loss (%s) and lets Enter recover in the same scrollback',
  async code => {
    seedWebappSession('test-token')
    vi.stubGlobal('WebSocket', Socket)
    installBrowserDesktopBridge()
    const id = createTerminal('/work')
    updateTerminalReviveBuffer(id, SAVED_HISTORY)
    render(<Harness id={id} />)
    await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
    const first = Socket.instances[0]
    act(() => {
      first.output('live output')
      first.close(code)
    })
    expect($terminals.get().find(tab => tab.id === id)?.reviveBuffer).toBe(SAVED_HISTORY)
    expect(screen.getByText('closed')).toBeTruthy()
    expect(emulator.output).toMatch(/disconnected.*Enter/i)
    const prior = emulator.output
    act(() => emulator.input('\r'))
    await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
    expect(Socket.instances).toHaveLength(2)
    expect(emulator.output.startsWith(prior)).toBe(true)
    expect(emulator.output).toContain('saved history')
    expect(emulator.output).toContain('live output')
    act(() => Socket.instances[1].close(4410, 'shell exited'))
    expect($terminals.get().some(tab => tab.id === id)).toBe(false)
  }
)

it.each(['attach', 'disconnect'])('cleans up the %s attempt before retry and ignores its late exit', async failure => {
  seedWebappSession('test-token')
  vi.stubGlobal('WebSocket', Socket)
  installBrowserDesktopBridge()
  const api = window.hermesDesktop!.terminal!
  const listeners = new Set<symbol>()
  const exits: Array<Parameters<typeof api.onExit>[1]> = []
  const onData = api.onData.bind(api)
  const onExit = api.onExit.bind(api)
  vi.spyOn(api, 'onData').mockImplementation((sid, callback) => {
    const key = Symbol()
    listeners.add(key)
    const unsubscribe = onData(sid, callback)

    return () => {
      listeners.delete(key)
      unsubscribe()
    }
  })
  vi.spyOn(api, 'onExit').mockImplementation((sid, callback) => {
    const key = Symbol()
    listeners.add(key)
    exits.push(callback)
    const unsubscribe = onExit(sid, callback)

    return () => {
      listeners.delete(key)
      unsubscribe()
    }
  })
  const dispose = vi.spyOn(api, 'dispose')

  if (failure === 'attach') {
    vi.spyOn(api, 'attach').mockRejectedValueOnce(new Error('attach failed'))
  }

  const id = createTerminal('/work')
  const view = render(<Harness id={id} />)

  if (failure === 'disconnect') {
    await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
    act(() => Socket.instances[0].close(1006))
  }

  await waitFor(() => expect(screen.getByText('closed')).toBeTruthy())
  expect(listeners.size).toBe(0)
  expect(dispose).toHaveBeenCalledTimes(1)
  expect(Socket.instances[0].readyState).toBe(3)
  act(() => emulator.input('\r'))
  await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
  expect(listeners.size).toBe(2)
  act(() => exits[0]({ code: 0, signal: null }))
  expect($terminals.get().some(tab => tab.id === id)).toBe(true)
  expect(screen.getByText('open')).toBeTruthy()
  act(() => exits[1]({ code: 0, signal: null }))
  expect($terminals.get().some(tab => tab.id === id)).toBe(false)
  expect(dispose).toHaveBeenCalledTimes(2) // Release the exited session before React unmounts.
  view.unmount()
  expect(dispose).toHaveBeenCalledTimes(2)
  expect(listeners.size).toBe(0)
  vi.restoreAllMocks()
})

it('replays server history once, suppresses parser replies, and detaches on remount', async () => {
  seedWebappSession('test-token')
  Socket.persistent = true
  vi.stubGlobal('WebSocket', Socket)
  installBrowserDesktopBridge()
  const api = window.hermesDesktop!.terminal
  const detach = vi.spyOn(api, 'detach')
  const dispose = vi.spyOn(api, 'dispose')
  const id = createTerminal('/work')
  updateTerminalReviveBuffer(id, SAVED_HISTORY)
  const firstView = render(<Harness id={id} />)
  await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
  expect(emulator.output).not.toContain('saved history')
  expect($terminals.get()[0].persistent).toBe(true)
  // Creation can already have a buffered DA/DSR query; an empty snapshot has
  // no marker, so only the first frame is conservative, not all future output.
  act(() => Socket.instances[0].output('startup history\u001b[6n'))
  expect(Socket.instances[0].sent).toEqual([])
  act(() => Socket.instances[0].output('live query\u001b[6n'))
  expect(Socket.instances[0].sent).toEqual(['\u001b[1;1R'])
  firstView.unmount()
  expect(detach).toHaveBeenCalledTimes(1)
  expect(dispose).not.toHaveBeenCalled()
  render(<Harness id={id} />)
  await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
  const second = Socket.instances[1]
  expect(second.url.searchParams.get('attach')).toBe('shell-1')
  act(() => second.output('history\u001b[6n'))
  expect(second.sent).toEqual(['\u001b[RESIZE:80;24]'])
  act(() => second.output('live query\u001b[6n'))
  expect(second.sent).toEqual(['\u001b[RESIZE:80;24]', '\u001b[1;1R'])
  act(() => second.close(4410))
  expect(screen.getByText('closed')).toBeTruthy()
  expect(emulator.output).toContain('Press Enter to create a new shell')
  expect($terminals.get().some(tab => tab.id === id)).toBe(true)
  expect(Socket.instances).toHaveLength(2)
  act(() => emulator.input('\r'))
  await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
  expect(Socket.instances[2].url.searchParams.has('attach')).toBe(false)
})

// Typing `exit` ends the host shell: the tab closes as it does in native Desktop, instead of
// staying open as an expired terminal.
it('closes a persistent browser tab when its host shell exits', async () => {
  seedWebappSession('test-token')
  Socket.persistent = true
  vi.stubGlobal('WebSocket', Socket)
  installBrowserDesktopBridge()
  const id = createTerminal('/work')
  render(<Harness id={id} />)
  await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
  act(() => Socket.instances[0].close(4410, 'Process exited'))
  expect($terminals.get().some(tab => tab.id === id)).toBe(false)
  expect(emulator.output).not.toContain('Press Enter to create a new shell')
})

// The shell exited while no view was attached (`exit`, then the tab was hidden or the page reloaded).
it('closes a persistent browser tab whose shell exited while it was detached', async () => {
  seedWebappSession('test-token')
  Socket.persistent = true
  vi.stubGlobal('WebSocket', Socket)
  installBrowserDesktopBridge()
  const id = createTerminal('/work')
  const firstView = render(<Harness id={id} />)
  await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
  firstView.unmount()
  Socket.refusal = [4410, 'Process exited']
  render(<Harness id={id} />)
  await waitFor(() => expect($terminals.get().some(tab => tab.id === id)).toBe(false))
  expect(Socket.instances[1].url.searchParams.get('attach')).toBe('shell-1')
  expect(emulator.output).not.toContain('Press Enter to create a new shell')
})

it('close others and close all terminate mounted and unmounted persisted shells', async () => {
  seedWebappSession('test-token')
  Socket.persistent = true
  vi.stubGlobal('WebSocket', Socket)
  installBrowserDesktopBridge()
  const api = window.hermesDesktop!.terminal
  const kept = createTerminal('/work')
  const other = createTerminal('/work')
  const session = await api.start({ cwd: '/work', restoreKey: other })
  await api.detach!(session.id)
  render(<Harness id={kept} />)
  await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
  act(() => closeOtherTerminals(kept))
  await waitFor(() => expect($terminals.get().map(tab => tab.id)).toEqual([kept]))
  act(() => closeAllTerminals())
  await waitFor(() => expect($terminals.get()).toEqual([]))
  const closes = Socket.instances.filter(socket => socket.url.searchParams.get('action') === 'close')
  expect(closes.map(socket => socket.url.searchParams.get('attach')).sort()).toEqual(['shell-1', 'shell-2'])
})

it('preserves saved tabs when page unload closes the browser socket', async () => {
  seedWebappSession('test-token')
  vi.stubGlobal('WebSocket', Socket)
  installBrowserDesktopBridge()
  const id = createTerminal('/work')
  updateTerminalReviveBuffer(id, SAVED_HISTORY)
  render(<Harness id={id} />)
  await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
  act(() => window.dispatchEvent(new Event('beforeunload')))
  expect(Socket.instances[0].readyState).toBe(3)
  expect($terminals.get().find(tab => tab.id === id)?.reviveBuffer).toBe(SAVED_HISTORY)
})

// Sign-out closes the tab's sockets with 4401. The terminal keeps its tab and scrollback and says
// how to get back in, instead of reconnecting against a refused credential.
it('keeps a signed-out tab and says to sign in again instead of reconnecting', async () => {
  seedWebappSession('test-token')
  Socket.persistent = true
  vi.stubGlobal('WebSocket', Socket)
  installBrowserDesktopBridge()
  const id = createTerminal('/work')
  render(<Harness id={id} />)
  await waitFor(() => expect(screen.getByText('open')).toBeTruthy())

  act(() => Socket.instances[0].close(4401, 'signed out'))

  await waitFor(() => expect(screen.getByText('closed')).toBeTruthy())
  expect(emulator.output).toMatch(/Signed out of Hermes\. Sign this browser in again/)
  expect(emulator.output).not.toMatch(/Reconnecting/)
  expect(Socket.instances).toHaveLength(1)
  expect($terminals.get().some(tab => tab.id === id)).toBe(true)
  expect($desktopBoot.get().error).toBe(WEBAPP_LAUNCH_REQUIRED)
  $desktopBoot.set({ ...$desktopBoot.get(), error: null, visible: false })
})

// Native Desktop's shell is a local process (node-pty over IPC). A failed spawn prints main's error
// line and leaves the tab closed: no retry hint, and Enter starts no second shell.
it("native Desktop: a failed start shows main's error and Enter does not start another shell", async () => {
  const start = vi.fn().mockRejectedValue(new Error('spawn /bin/nope ENOENT'))

  delete window.document.documentElement.dataset.hermesDesktopHost
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    terminal: {
      attach: vi.fn(),
      cwd: vi.fn(async () => null),
      dispose: vi.fn(async () => true),
      onData: vi.fn(() => () => {}),
      onExit: vi.fn(() => () => {}),
      resize: vi.fn(async () => true),
      start,
      write: vi.fn(async () => true)
    }
  }
  const id = createTerminal('/work')
  render(<Harness id={id} />)

  await waitFor(() => expect(emulator.output).toContain('Terminal failed to start: spawn /bin/nope ENOENT\r\n'))
  expect(screen.getByText('closed')).toBeTruthy()
  expect(emulator.output).not.toMatch(/Press Enter/)

  act(() => emulator.input('\r'))
  await act(async () => {})
  expect(start).toHaveBeenCalledTimes(1)
  expect(screen.getByText('closed')).toBeTruthy()
})

// An idle tab never re-saves its scrollback (#61572): the live buffer holds the replayed history
// plus a fresh prompt. PowerShell (PSReadLine) and fish ask for the cursor position at startup, and
// xterm's reply reaches the shell through onData like typed input; counting that reply as activity
// re-saved those shells' idle tabs on every launch. Only keys, typing, composition, paste, drop and
// mouse input with mouse tracking on count.
it("native Desktop: xterm's reply to the shell's cursor query is not activity; typing is", async () => {
  let shellOutput: ((data: string) => void) | null = null
  const write = vi.fn(async () => true)

  delete window.document.documentElement.dataset.hermesDesktopHost
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    terminal: {
      attach: vi.fn(async () => true),
      cwd: vi.fn(async () => null),
      dispose: vi.fn(async () => true),
      onData: vi.fn((_id: string, callback: (data: string) => void) => {
        shellOutput = callback

        return () => {}
      }),
      onExit: vi.fn(() => () => {}),
      resize: vi.fn(async () => true),
      start: vi.fn(async () => ({ id: 'pty-1', shell: 'pwsh.exe' })),
      write
    }
  }
  const id = createTerminal('/work')
  updateTerminalReviveBuffer(id, SAVED_HISTORY)
  const { container } = render(<Harness id={id} />)
  await waitFor(() => expect(shellOutput).not.toBeNull())
  const saved = () => $terminals.get().find(tab => tab.id === id)?.reviveBuffer

  act(() => shellOutput!('PS /work> \u001b[6n'))

  expect(write).toHaveBeenCalledWith('pty-1', '\u001b[1;1R')
  expect(saved()).toBe(SAVED_HISTORY)

  act(() => {
    container.querySelector('div')!.dispatchEvent(new Event('beforeinput'))
    shellOutput!('make\r\nbuilt ok\r\n')
  })

  await waitFor(() => expect(saved()).toContain('$ make'), { timeout: 2000 })
})
