import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { DiffusionCanvas } from './image-generation-placeholder'

let root: Root | null = null
let container: HTMLDivElement | null = null
let windowStateCallback: ((payload: { isMinimized?: boolean; isVisible?: boolean }) => void) | null = null

function render() {
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)

  act(() => {
    root!.render(<DiffusionCanvas />)
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
    pending: () => frames.size,
    request
  }
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

describe('DiffusionCanvas scheduling', () => {
  beforeEach(() => {
    ;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    installWindowStateBridge()
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      setTransform: vi.fn()
    } as unknown as CanvasRenderingContext2D)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('cancels its loop while inactive and resumes only when the window is observable', () => {
    const raf = installRaf()

    render()
    expect(raf.request).toHaveBeenCalledTimes(1)
    expect(raf.pending()).toBe(1)

    act(() => {
      window.dispatchEvent(new Event('blur'))
    })
    expect(raf.pending()).toBe(0)

    act(() => {
      window.dispatchEvent(new Event('focus'))
    })
    expect(raf.pending()).toBe(1)

    act(() => {
      windowStateCallback?.({ isMinimized: true, isVisible: false })
    })
    expect(raf.pending()).toBe(0)

    cleanup()
    act(() => {
      window.dispatchEvent(new Event('focus'))
    })
    expect(raf.pending()).toBe(0)
  })
})
