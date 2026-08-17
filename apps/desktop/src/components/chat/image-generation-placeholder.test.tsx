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
    request,
    runNext: (now: number) => {
      const next = frames.entries().next().value

      if (!next) {
        throw new Error('No pending RAF')
      }

      const [id, callback] = next
      frames.delete(id)
      callback(now)
    }
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

/** 2D-context stand-in whose every method is a lazily-created spy — the draw
 *  path touches gradients, transforms, and fills, and this keeps the mock
 *  honest without enumerating the full CanvasRenderingContext2D surface. */
function installDrawableContext() {
  const clearRect = vi.fn()
  const gradient = { addColorStop: vi.fn() }
  const fns = new Map<PropertyKey, ReturnType<typeof vi.fn>>()
  const ctx = new Proxy(
    {},
    {
      get(_target, prop) {
        if (prop === 'clearRect') {
          return clearRect
        }

        if (prop === 'createLinearGradient' || prop === 'createRadialGradient') {
          return () => gradient
        }

        if (!fns.has(prop)) {
          fns.set(prop, vi.fn())
        }

        return fns.get(prop)
      },
      set() {
        return true
      }
    }
  )

  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctx as unknown as CanvasRenderingContext2D)

  return { clearRect }
}

describe('DiffusionCanvas frame budget', () => {
  beforeEach(() => {
    ;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    installWindowStateBridge()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('paints at most ~15fps: frames inside the interval reschedule without redrawing', () => {
    const raf = installRaf()
    const { clearRect } = installDrawableContext()

    render()
    expect(raf.pending()).toBe(1)

    act(() => {
      raf.runNext(1000) // first paint
    })
    const paintsAfterFirst = clearRect.mock.calls.length
    expect(paintsAfterFirst).toBeGreaterThan(0)

    act(() => {
      raf.runNext(1010) // 10ms later — inside the 1000/15 ≈ 66.7ms budget
    })
    expect(clearRect.mock.calls.length).toBe(paintsAfterFirst)
    expect(raf.pending()).toBe(1) // loop kept alive, just no repaint

    act(() => {
      raf.runNext(1080) // past the budget — repaints
    })
    expect(clearRect.mock.calls.length).toBeGreaterThan(paintsAfterFirst)
  })

  it('caps concurrent animated instances: extras draw one static frame and skip the loop', () => {
    const raf = installRaf()
    installDrawableContext()

    const containers: HTMLDivElement[] = []
    const roots: Root[] = []

    act(() => {
      for (let i = 0; i < 3; i++) {
        const el = document.createElement('div')
        document.body.append(el)
        containers.push(el)
        const r = createRoot(el)
        roots.push(r)
        r.render(<DiffusionCanvas />)
      }
    })

    // Two animated loops pending; the third instance drew statically and
    // scheduled nothing.
    expect(raf.pending()).toBe(2)

    act(() => {
      for (const r of roots) {
        r.unmount()
      }
    })
    for (const el of containers) {
      el.remove()
    }

    // Counter released on unmount — a fresh mount animates again.
    const el = document.createElement('div')
    document.body.append(el)
    const r = createRoot(el)

    act(() => {
      r.render(<DiffusionCanvas />)
    })
    expect(raf.pending()).toBe(1)

    act(() => {
      r.unmount()
    })
    el.remove()
  })
})
