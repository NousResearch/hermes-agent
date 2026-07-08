import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/pet', () => {
  const listeners = new Set<(state: string) => void>()

  return {
    $petState: {
      get: () => 'idle',
      listen: (callback: (state: string) => void) => {
        listeners.add(callback)

        return () => {
          listeners.delete(callback)
        }
      }
    }
  }
})

import { PetSprite } from './pet-sprite'

const INFO = {
  enabled: true,
  frameH: 16,
  frameW: 16,
  framesPerState: 2,
  loopMs: 120,
  scale: 1,
  spritesheetBase64: 'stub',
  stateRows: ['idle']
}

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

describe('PetSprite RAF scheduling', () => {
  beforeEach(() => {
    ;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    vi.useFakeTimers()
    setVisibility(false)
    installWindowStateBridge()
    vi.stubGlobal(
      'Image',
      class extends EventTarget {
        complete = true
        naturalWidth = 16
        src = ''
      } as unknown as typeof Image
    )
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      clearRect: vi.fn(),
      drawImage: vi.fn(),
      imageSmoothingEnabled: false
    } as unknown as CanvasRenderingContext2D)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    setVisibility(false)
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('sleeps between visible sprite frames instead of chaining RAFs', () => {
    const raf = installRaf()

    render(<PetSprite info={INFO} />)

    expect(raf.request).toHaveBeenCalledTimes(1)

    act(() => {
      raf.runNext(0)
    })

    expect(raf.request).toHaveBeenCalledTimes(1)
    expect(raf.pending()).toBe(0)
    expect(vi.getTimerCount()).toBe(1)

    act(() => {
      vi.advanceTimersByTime(60)
    })

    expect(raf.request).toHaveBeenCalledTimes(2)
  })

  it('cancels pending RAF work while the Electron window is paused and resumes when visible', () => {
    const raf = installRaf()

    render(<PetSprite info={INFO} />)

    expect(raf.request).toHaveBeenCalledTimes(1)

    act(() => {
      windowStateCallback?.({ isMinimized: true, isVisible: false })
    })

    expect(raf.cancel).toHaveBeenCalledTimes(1)
    expect(raf.pending()).toBe(0)

    act(() => {
      windowStateCallback?.({ isMinimized: false, isVisible: true })
    })

    expect(raf.request).toHaveBeenCalledTimes(2)
  })
})
