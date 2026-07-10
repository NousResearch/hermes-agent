import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PersistentTerminal, TerminalSlot } from './persistent'

vi.mock('./terminals', () => ({
  ensureTerminal: vi.fn()
}))

vi.mock('./workspace', () => ({
  TerminalWorkspace: () => <div data-testid="terminal-workspace" />
}))

let resizeObserverCallback: ResizeObserverCallback | null = null
let mutationObserverCallback: MutationCallback | null = null
let root: Root | null = null
let container: HTMLDivElement | null = null
let windowStateCallback: ((payload: { isMinimized?: boolean; isVisible?: boolean }) => void) | null = null

function render(ui: ReactNode) {
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)

  act(() => {
    root!.render(ui)
  })
}

function cleanup() {
  if (root) {
    act(() => {
      root!.unmount()
    })
  }

  container?.remove()
  root = null
  container = null
}

function setVisibility(hidden: boolean) {
  Object.defineProperty(document, 'hidden', { configurable: true, value: hidden })
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: hidden ? 'hidden' : 'visible' })
}

function installWindowStateBridge() {
  windowStateCallback = null
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      onWindowStateChanged: vi.fn((callback: typeof windowStateCallback) => {
        windowStateCallback = callback

        return () => {
          if (windowStateCallback === callback) {
            windowStateCallback = null
          }
        }
      })
    }
  })
}

function rect(top: number, left: number, width: number, height: number): DOMRect {
  return {
    bottom: top + height,
    height,
    left,
    right: left + width,
    top,
    width,
    x: left,
    y: top,
    toJSON: () => ({})
  } as DOMRect
}

function installRaf() {
  let nextId = 1
  const frames = new Map<number, FrameRequestCallback>()
  const request = vi.fn((callback: FrameRequestCallback) => {
    const id = nextId++
    frames.set(id, callback)

    return id
  })
  const cancel = vi.fn((id: number) => {
    frames.delete(id)
  })

  Object.defineProperty(window, 'requestAnimationFrame', { configurable: true, value: request })
  Object.defineProperty(window, 'cancelAnimationFrame', { configurable: true, value: cancel })

  return {
    cancel,
    pending: () => frames.size,
    request,
    runNext: () => {
      const next = frames.entries().next().value

      if (!next) {
        throw new Error('No pending RAF')
      }

      const [id, callback] = next
      frames.delete(id)
      callback(0)
    }
  }
}

function Harness() {
  return (
    <>
      <TerminalSlot className="slot" />
      <PersistentTerminal onAddSelectionToChat={() => undefined} />
    </>
  )
}

describe('PersistentTerminal rect tracking', () => {
  beforeEach(() => {
    ;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    setVisibility(false)
    installWindowStateBridge()
    resizeObserverCallback = null
    mutationObserverCallback = null
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(callback: ResizeObserverCallback) {
          resizeObserverCallback = callback
        }

        disconnect = vi.fn()
        observe = vi.fn()
        unobserve = vi.fn()
      } as unknown as typeof ResizeObserver
    )
    vi.stubGlobal(
      'MutationObserver',
      class {
        constructor(callback: MutationCallback) {
          mutationObserverCallback = callback
        }

        disconnect = vi.fn()
        observe = vi.fn()
        takeRecords = vi.fn(() => [])
      } as unknown as typeof MutationObserver
    )
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    setVisibility(false)
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('settles after rect changes instead of polling forever', () => {
    const raf = installRaf()
    let currentRect = rect(10, 20, 200, 100)
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(() => currentRect)

    render(<Harness />)

    expect(raf.request).toHaveBeenCalledTimes(1)

    act(() => {
      raf.runNext()
    })

    expect(raf.request).toHaveBeenCalledTimes(1)
    expect(raf.pending()).toBe(0)

    currentRect = rect(12, 24, 220, 120)
    act(() => {
      resizeObserverCallback?.([], {} as ResizeObserver)
    })

    expect(raf.request).toHaveBeenCalledTimes(2)

    act(() => {
      raf.runNext()
    })

    expect(raf.request).toHaveBeenCalledTimes(3)

    act(() => {
      raf.runNext()
    })

    expect(raf.request).toHaveBeenCalledTimes(3)
    expect(raf.pending()).toBe(0)
  })

  it('remeasures when layout moves the slot without resizing it', () => {
    const raf = installRaf()
    let currentRect = rect(10, 20, 200, 100)
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(() => currentRect)

    render(<Harness />)

    act(() => {
      raf.runNext()
    })

    const overlay = container!.lastElementChild as HTMLElement
    expect(overlay.style.top).toBe('10px')
    expect(overlay.style.left).toBe('20px')
    expect(raf.pending()).toBe(0)

    currentRect = rect(32, 48, 200, 100)
    act(() => {
      mutationObserverCallback?.([], {} as MutationObserver)
    })

    expect(raf.request).toHaveBeenCalledTimes(2)

    act(() => {
      raf.runNext()
    })

    expect(overlay.style.top).toBe('32px')
    expect(overlay.style.left).toBe('48px')

    act(() => {
      raf.runNext()
    })

    expect(raf.pending()).toBe(0)
  })

  it('does not schedule rect RAFs while the Electron window is paused, then resumes when visible', () => {
    const raf = installRaf()
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(rect(10, 20, 200, 100))

    render(<Harness />)

    expect(raf.request).toHaveBeenCalledTimes(1)

    act(() => {
      windowStateCallback?.({ isMinimized: true, isVisible: false })
    })

    expect(raf.cancel).toHaveBeenCalledTimes(1)
    expect(raf.pending()).toBe(0)

    act(() => {
      resizeObserverCallback?.([], {} as ResizeObserver)
    })

    expect(raf.request).toHaveBeenCalledTimes(1)

    act(() => {
      windowStateCallback?.({ isMinimized: false, isVisible: true })
    })

    expect(raf.request).toHaveBeenCalledTimes(2)
  })
})
