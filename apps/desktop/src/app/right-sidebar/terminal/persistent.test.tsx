import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { hiddenPaneProps, PANE_HIDDEN_ATTR } from '@/components/pane-shell/pane-visibility'
import { $paneStates } from '@/store/panes'
import { reactRoot } from '@/test/react-root'

import { installWindowStateBridge, setDocumentHidden, type WindowStateBridge } from '../../../test/window-state'
import { $terminalTakeover } from '../store'

import { PersistentTerminal, TerminalSlot } from './persistent'

vi.mock('../store', async () => ({
  $terminalTakeover: (await import('nanostores')).atom(false)
}))

vi.mock('./terminals', () => ({
  ensureTerminal: vi.fn()
}))

vi.mock('./workspace', () => ({
  TerminalWorkspace: () => <div data-testid="terminal-workspace" />
}))

let resizeObserverCallback: ResizeObserverCallback | null = null
let mutationObserverCallback: MutationCallback | null = null
let mutationObserveCalls: Array<{ options?: MutationObserverInit; target: Node }> = []
const mount = reactRoot()
let windowState: WindowStateBridge

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

function HiddenPaneHarness({ hidden }: { hidden: boolean }) {
  return (
    <>
      <div {...hiddenPaneProps(hidden)}>
        <TerminalSlot className="slot" />
      </div>
      <PersistentTerminal onAddSelectionToChat={() => undefined} />
    </>
  )
}

describe('PersistentTerminal rect tracking', () => {
  beforeEach(() => {
    ;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    setDocumentHidden(false)
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    windowState = installWindowStateBridge()
    resizeObserverCallback = null
    mutationObserverCallback = null
    mutationObserveCalls = []
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
        observe = vi.fn((target: Node, options?: MutationObserverInit) => {
          mutationObserveCalls.push({ options, target })
        })
        takeRecords = vi.fn(() => [])
      } as unknown as typeof MutationObserver
    )
  })

  afterEach(() => {
    mount.unmount()
    $terminalTakeover.set(false)
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    setDocumentHidden(false)
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('settles after rect changes instead of polling forever', () => {
    const raf = installRaf()
    let currentRect = rect(10, 20, 200, 100)
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(() => currentRect)

    mount.render(<Harness />)

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

    mount.render(<Harness />)

    expect(mutationObserveCalls.length).toBeGreaterThan(0)
    expect(mutationObserveCalls.every(call => call.options?.subtree === false)).toBe(true)

    act(() => {
      raf.runNext()
    })

    const overlay = mount.container!.lastElementChild as HTMLElement
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

  it('remeasures from an explicit pane-layout state change', () => {
    const raf = installRaf()
    const before = $paneStates.get()
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(rect(10, 20, 200, 100))

    mount.render(<Harness />)
    raf.runNext()
    expect(raf.pending()).toBe(0)

    act(() => {
      $paneStates.set({ ...before, __terminal_rect_test__: { open: true } })
    })

    expect(raf.pending()).toBe(1)

    act(() => {
      raf.runNext()
      $paneStates.set(before)
    })
  })

  it('does not schedule rect RAFs while the Electron window is paused, then resumes when visible', () => {
    const raf = installRaf()
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(rect(10, 20, 200, 100))

    mount.render(<Harness />)

    expect(raf.request).toHaveBeenCalledTimes(1)

    act(() => {
      windowState.emit({ isMinimized: true, isVisible: false })
    })

    expect(raf.cancel).toHaveBeenCalledTimes(1)
    expect(raf.pending()).toBe(0)

    act(() => {
      resizeObserverCallback?.([], {} as ResizeObserver)
    })

    expect(raf.request).toHaveBeenCalledTimes(1)

    act(() => {
      windowState.emit({ isMinimized: false, isVisible: true })
    })

    expect(raf.request).toHaveBeenCalledTimes(2)
  })

  it('suspends while unfocused and cancels every callback on unmount', () => {
    const raf = installRaf()
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(rect(10, 20, 200, 100))

    mount.render(<Harness />)
    expect(raf.pending()).toBe(1)

    act(() => window.dispatchEvent(new Event('blur')))
    expect(raf.pending()).toBe(0)

    act(() => window.dispatchEvent(new Event('focus')))
    expect(raf.pending()).toBe(1)

    mount.unmount()
    expect(raf.pending()).toBe(0)

    act(() => {
      window.dispatchEvent(new Event('focus'))
      resizeObserverCallback?.([], {} as ResizeObserver)
      mutationObserverCallback?.([], {} as MutationObserver)
    })
    expect(raf.pending()).toBe(0)
  })

  it('does not schedule an initial frame when mounted while unfocused', () => {
    const raf = installRaf()
    vi.mocked(document.hasFocus).mockReturnValue(false)
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(rect(10, 20, 200, 100))

    mount.render(<Harness />)

    expect(raf.request).not.toHaveBeenCalled()

    act(() => window.dispatchEvent(new Event('focus')))

    expect(raf.pending()).toBe(1)
  })

  it('hides the overlay but keeps its workspace mounted when the terminal tab becomes inactive', () => {
    const raf = installRaf()
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(rect(10, 20, 200, 100))
    $terminalTakeover.set(true)

    mount.render(<HiddenPaneHarness hidden={false} />)

    const overlay = mount.container!.lastElementChild as HTMLElement
    const workspace = mount.container!.querySelector('[data-testid="terminal-workspace"]')

    expect(overlay.style.visibility).toBe('visible')
    expect(overlay.style.opacity).toBe('1')
    expect(overlay.style.pointerEvents).toBe('auto')
    expect(workspace).not.toBeNull()
    expect(mutationObserveCalls.some(call => call.options?.attributeFilter?.includes(PANE_HIDDEN_ATTR))).toBe(true)

    // Let the initial changed-rect follow-up settle. The hidden-pane transition
    // must wake the observer from an otherwise idle state.
    act(() => {
      raf.runNext()
    })
    expect(raf.pending()).toBe(0)

    act(() => {
      mount.root!.render(<HiddenPaneHarness hidden />)
      mutationObserverCallback?.([], {} as MutationObserver)
    })
    expect(raf.pending()).toBe(1)

    act(() => {
      raf.runNext()
    })

    expect(overlay.style.visibility).toBe('hidden')
    expect(overlay.style.opacity).toBe('0')
    expect(overlay.style.pointerEvents).toBe('none')
    expect(mount.container!.querySelector('[data-testid="terminal-workspace"]')).toBe(workspace)

    act(() => {
      raf.runNext()
    })
    expect(raf.pending()).toBe(0)

    act(() => {
      mount.root!.render(<HiddenPaneHarness hidden={false} />)
      mutationObserverCallback?.([], {} as MutationObserver)
    })

    act(() => {
      raf.runNext()
    })

    expect(overlay.style.visibility).toBe('visible')
    expect(overlay.style.opacity).toBe('1')
    expect(overlay.style.pointerEvents).toBe('auto')
    expect(mount.container!.querySelector('[data-testid="terminal-workspace"]')).toBe(workspace)
  })
})
