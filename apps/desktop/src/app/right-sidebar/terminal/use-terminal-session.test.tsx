import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'

import { $terminals, createTerminal, updateTerminalReviveBuffer } from './terminals'
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
vi.mock('@xterm/addon-serialize', () => ({ SerializeAddon: class {} }))
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
  readyState = 1
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  constructor() {
    Socket.instances.push(this)
    queueMicrotask(() => this.onmessage?.({ data: '\u0000HERMES_TERMINAL_META:{"shell":"fish"}' } as MessageEvent))
  }
  send() {}
  close(code = 1000, reason = '') {
    this.readyState = 3
    this.onclose?.(new CloseEvent('close', { code, reason }))
  }
  output(data: string) {
    this.onmessage?.({ data } as MessageEvent)
  }
}

function Harness({ id }: { id: string }) {
  const { hostRef, status } = useTerminalSession({
    id,
    cwd: '/work',
    active: false,
    reviveBuffer: 'saved history',
    onAddSelectionToChat: () => {}
  })

  return (
    <>
      <span>{status}</span>
      <div ref={hostRef} />
    </>
  )
}

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(window, 'hermesDesktop')
  Reflect.deleteProperty(window, '__HERMES_SESSION_TOKEN__')
  $terminals.set([])
  Socket.instances = []
  emulator.output = ''
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it.each([1006, 1000])('retains the saved tab on transport loss (%s) and lets Enter recover in the same scrollback', async code => {
  Object.assign(window, { __HERMES_SESSION_TOKEN__: 'test-token' })
  vi.stubGlobal('WebSocket', Socket)
  installBrowserDesktopBridge()
  const id = createTerminal('/work')
  updateTerminalReviveBuffer(id, 'saved history')
  render(<Harness id={id} />)
  await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
  const first = Socket.instances[0]
  act(() => {
    first.output('live output')
    first.close(code)
  })
  expect($terminals.get().find(tab => tab.id === id)?.reviveBuffer).toBe('saved history')
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
})

it.each(['attach', 'disconnect'])('cleans up the %s attempt before retry and ignores its late exit', async failure => {
  Object.assign(window, { __HERMES_SESSION_TOKEN__: 'test-token' })
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

    return () => { listeners.delete(key); unsubscribe() }
  })
  vi.spyOn(api, 'onExit').mockImplementation((sid, callback) => {
    const key = Symbol()
    listeners.add(key)
    exits.push(callback)
    const unsubscribe = onExit(sid, callback)

    return () => { listeners.delete(key); unsubscribe() }
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
  view.unmount()
  expect(listeners.size).toBe(0)
  vi.restoreAllMocks()
})

it('preserves saved tabs when page unload closes the browser socket', async () => {
  Object.assign(window, { __HERMES_SESSION_TOKEN__: 'test-token' })
  vi.stubGlobal('WebSocket', Socket)
  installBrowserDesktopBridge()
  const id = createTerminal('/work')
  updateTerminalReviveBuffer(id, 'saved history')
  render(<Harness id={id} />)
  await waitFor(() => expect(screen.getByText('open')).toBeTruthy())
  act(() => window.dispatchEvent(new Event('beforeunload')))
  expect(Socket.instances[0].readyState).toBe(3)
  expect($terminals.get().find(tab => tab.id === id)?.reviveBuffer).toBe('saved history')
})
