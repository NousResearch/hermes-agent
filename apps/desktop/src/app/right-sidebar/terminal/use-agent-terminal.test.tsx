import { act, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useAgentTerminal } from './use-agent-terminal'

const xterm = vi.hoisted(() => ({
  attachCustomKeyEventHandler: vi.fn(),
  clearSelection: vi.fn(),
  dispose: vi.fn(),
  focus: vi.fn(),
  getSelection: vi.fn(() => ''),
  loadAddon: vi.fn(),
  onSelectionChange: vi.fn(() => ({ dispose: vi.fn() })),
  open: vi.fn(),
  refresh: vi.fn(),
  write: vi.fn()
}))

const terminalRegistrations = vi.hoisted(() => ({
  makeTerminalReader: vi.fn(() => vi.fn()),
  registerReader: vi.fn(() => vi.fn()),
  registerWriter: vi.fn(() => vi.fn())
}))

const webglContextLossHandlers = vi.hoisted(() => [] as Array<() => void>)

vi.mock('@xterm/xterm', () => ({
  Terminal: class {
    readonly buffer = { active: {} }
    readonly rows = 24
    readonly unicode = { activeVersion: '6' }
    options: Record<string, unknown>

    constructor(options: Record<string, unknown>) {
      this.options = { ...options }
    }

    attachCustomKeyEventHandler = xterm.attachCustomKeyEventHandler
    clearSelection = xterm.clearSelection
    dispose = xterm.dispose
    focus = xterm.focus
    getSelection = xterm.getSelection
    loadAddon = xterm.loadAddon
    onSelectionChange = xterm.onSelectionChange
    open = xterm.open
    refresh = xterm.refresh
    write = xterm.write
  }
}))

vi.mock('@xterm/addon-fit', () => ({
  FitAddon: class {
    fit = vi.fn()
  }
}))

vi.mock('@xterm/addon-unicode11', () => ({
  Unicode11Addon: class {}
}))

vi.mock('@xterm/addon-web-links', () => ({
  WebLinksAddon: class {}
}))

vi.mock('@xterm/addon-webgl', () => ({
  WebglAddon: class {
    clearTextureAtlas = vi.fn()
    dispose = vi.fn()
    onContextLoss = vi.fn((handler: () => void) => {
      webglContextLossHandlers.push(handler)
    })
  }
}))

vi.mock('@/components/ui/copy-button', () => ({
  writeClipboardText: vi.fn()
}))

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: vi.fn()
}))

vi.mock('@/themes/context', () => ({
  useTheme: () => ({
    renderedMode: 'dark',
    theme: { terminal: {} },
    themeName: 'test'
  })
}))

vi.mock('./agent-terminal-stream', () => ({
  registerAgentTerminalWriter: terminalRegistrations.registerWriter
}))

vi.mock('./buffer', () => ({
  makeTerminalReader: terminalRegistrations.makeTerminalReader,
  registerTerminalReader: terminalRegistrations.registerReader
}))

function Harness() {
  const { hostRef } = useAgentTerminal({ active: false, id: 'agent-tab', procId: 'proc-1' })

  return <div ref={hostRef} />
}

describe('useAgentTerminal', () => {
  let resolveFontLoad!: (faces: FontFace[]) => void
  let resizeObserverConstructor = vi.fn<() => void>()

  beforeEach(() => {
    const pendingFontLoad = new Promise<FontFace[]>(resolve => {
      resolveFontLoad = resolve
    })

    Object.defineProperty(globalThis.document, 'fonts', {
      configurable: true,
      value: { load: vi.fn(() => pendingFontLoad) }
    })

    resizeObserverConstructor = vi.fn<() => void>()
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor() {
          resizeObserverConstructor()
        }

        disconnect = vi.fn()
        observe = vi.fn()
        unobserve = vi.fn()
      } as unknown as typeof ResizeObserver
    )
  })

  afterEach(() => {
    vi.clearAllMocks()
    vi.unstubAllGlobals()
    Reflect.deleteProperty(globalThis.document, 'fonts')
  })

  it('unmounts safely while initial font preparation is pending', async () => {
    const { unmount } = render(<Harness />)

    await waitFor(() => expect(globalThis.document.fonts.load).toHaveBeenCalled())

    expect(() => unmount()).not.toThrow()
    expect(xterm.dispose).toHaveBeenCalledOnce()
    expect(resizeObserverConstructor).not.toHaveBeenCalled()

    await act(async () => {
      resolveFontLoad([])
      await Promise.resolve()
    })

    expect(xterm.open).not.toHaveBeenCalled()
    expect(resizeObserverConstructor).not.toHaveBeenCalled()
    expect(terminalRegistrations.registerWriter).not.toHaveBeenCalled()
    expect(terminalRegistrations.registerReader).not.toHaveBeenCalled()
  })

  it('repaints buffered rows through the DOM renderer after a WebGL context loss', async () => {
    webglContextLossHandlers.length = 0
    render(<Harness />)

    await act(async () => {
      resolveFontLoad([])
      await Promise.resolve()
    })

    expect(xterm.open).toHaveBeenCalled()
    expect(webglContextLossHandlers.length).toBeGreaterThan(0)

    xterm.refresh.mockClear()

    // The addon fires context loss; the hook must dispose the dead renderer,
    // drop the ref, and force a repaint so the viewport doesn't stay black
    // while the buffer is intact.
    const handler = webglContextLossHandlers[webglContextLossHandlers.length - 1]
    expect(() => handler()).not.toThrow()
    expect(xterm.refresh).toHaveBeenCalledWith(0, 23)
  })

  it('retries the mount once a host rendered disconnected joins the document', async () => {
    // The #118004 strand: the pane shell can render the host before it is
    // connected to the document, so the font wait's isCurrent() goes false at
    // an await boundary and resolves null — the old code returned silently
    // and the pane stayed blank forever (no open, no stream attach). The
    // mount must re-arm and run as soon as the host connects.
    const frames = new Map<number, FrameRequestCallback>()
    let nextFrameId = 1

    const request = vi.fn((callback: FrameRequestCallback) => {
      const id = nextFrameId++

      frames.set(id, callback)

      return id
    })

    vi.stubGlobal('requestAnimationFrame', request)
    vi.stubGlobal(
      'cancelAnimationFrame',
      vi.fn((id: number) => frames.delete(id))
    )

    // The effect runs while the container is still attached (the font wait
    // starts against a connected host), then the container is detached so
    // isCurrent() goes false at the warm() await boundary — the same race as
    // a pane shell that renders its host before the document connects it.
    const mount = render(<Harness />, {
      container: globalThis.document.body.appendChild(globalThis.document.createElement('div'))
    })

    const host = mount.container.querySelector('div')!

    // Disconnect the container so host.isConnected is false when the font
    // promise settles.
    mount.container.remove()

    await act(async () => {
      resolveFontLoad([])
      await Promise.resolve()
    })

    expect(host.isConnected).toBe(false)
    expect(xterm.open).not.toHaveBeenCalled()
    expect(terminalRegistrations.registerWriter).not.toHaveBeenCalled()

    // The watch is armed: reconnecting the host and draining the frame loop
    // retries the font wait (fonts resolve immediately now) and mounts.
    globalThis.document.body.appendChild(mount.container)

    await act(async () => {
      while (frames.size > 0) {
        const next = frames.entries().next().value as [number, FrameRequestCallback]
        frames.delete(next[0])
        next[1](0)
      }

      await Promise.resolve()
    })

    expect(xterm.open).toHaveBeenCalledWith(host)
    expect(terminalRegistrations.registerWriter).toHaveBeenCalled()

    // Tearing down while the watch is still armed must cancel it cleanly.
    mount.unmount()
    expect(() => {
      while (frames.size > 0) {
        const next = frames.entries().next().value as [number, FrameRequestCallback]
        frames.delete(next[0])
        next[1](0)
      }
    }).not.toThrow()
    expect(xterm.dispose).toHaveBeenCalled()
  })
})
