import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/pet', () => {
  const listeners = new Set<(state: string) => void>()

  return {
    $petSpriteTarget: {
      get: () => ({ state: 'idle' }),
      listen: (callback: (state: string) => void) => {
        listeners.add(callback)

        return () => {
          listeners.delete(callback)
        }
      }
    }
  }
})

import type { PetInfo } from '@/store/pet'
import { reactRoot } from '@/test/react-root'

import { installWindowStateBridge, setDocumentHidden, type WindowStateBridge } from '../../test/window-state'

import { roamHopRow } from './pet-rows'
import { PetSprite, ROW_HOLD_MS } from './pet-sprite'

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

const mount = reactRoot()
let windowState: WindowStateBridge
let drawImage: ReturnType<typeof vi.fn>

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
    setDocumentHidden(false)
    Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: 1 })
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    windowState = installWindowStateBridge()
    vi.stubGlobal(
      'Image',
      class extends EventTarget {
        complete = true
        naturalWidth = 16
        src = ''
      } as unknown as typeof Image
    )
    drawImage = vi.fn()
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      clearRect: vi.fn(),
      drawImage,
      imageSmoothingEnabled: false
    } as unknown as CanvasRenderingContext2D)
  })

  afterEach(() => {
    mount.unmount()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    setDocumentHidden(false)
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('sleeps between visible sprite frames instead of chaining RAFs', () => {
    const raf = installRaf()

    mount.render(<PetSprite info={INFO} />)

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

  const CYCLE = ['idle', 'running', 'dancing', 'juggling']
  const RUN = { rows: { run: ['dancing', 'juggling'] } }
  const FAILED = { rows: { failed: ['crying'] } }

  // [case, stateRows, behaviors, state, wall clock, row drawn]
  it.each([
    ['draws the row a behavior points a state at', ['idle', 'failed', 'review', 'crying'], FAILED, 'failed', 0, 3],
    ['ignores a behavior row the sheet lacks', ['idle', 'failed'], FAILED, 'failed', 0, 1],
    ['cycles listed rows by wall-clock slot (slot 3)', CYCLE, RUN, 'run', ROW_HOLD_MS * 3, 3],
    ['cycles listed rows by wall-clock slot (slot 4)', CYCLE, RUN, 'run', ROW_HOLD_MS * 4, 2],
    ["holds each row for the state's holdSeconds", CYCLE, { ...RUN, holdSeconds: { run: 5 } }, 'run', 5_000, 3]
  ] as const)('%s', (_case, stateRows, behaviors, state, now, row) => {
    const raf = installRaf()
    vi.setSystemTime(now)

    mount.render(<PetSprite info={{ ...INFO, stateRows, behaviors } as PetInfo} stateOverride={state} />)
    act(() => raf.runNext(0))

    expect(drawImage).toHaveBeenLastCalledWith(expect.anything(), 0, row * 16, 16, 16, 0, 0, 16, 16)
  })

  it('uses a DPR-sized backing store while preserving the CSS footprint', () => {
    Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: 2 })
    const raf = installRaf()

    mount.render(<PetSprite info={INFO} />)

    const canvas = mount.container?.querySelector('canvas')

    expect(canvas).not.toBeNull()
    expect(canvas?.width).toBe(32)
    expect(canvas?.height).toBe(32)
    expect(canvas?.style.width).toBe('16px')
    expect(canvas?.style.height).toBe('16px')

    act(() => {
      raf.runNext(0)
    })

    expect(drawImage).toHaveBeenCalledWith(expect.anything(), 0, 0, 16, 16, 0, 0, 32, 32)
  })

  it('cancels pending RAF work while the Electron window is paused and resumes when visible', () => {
    const raf = installRaf()

    mount.render(<PetSprite info={INFO} />)

    expect(raf.request).toHaveBeenCalledTimes(1)

    act(() => {
      windowState.emit({ isMinimized: true, isVisible: false })
    })

    expect(raf.cancel).toHaveBeenCalledTimes(1)
    expect(raf.pending()).toBe(0)

    act(() => {
      windowState.emit({ isMinimized: false, isVisible: true })
    })

    expect(raf.request).toHaveBeenCalledTimes(2)
  })

  it('keeps animating while unfocused and leaves no work after unmount', () => {
    const raf = installRaf()

    mount.render(<PetSprite info={INFO} />)

    act(() => window.dispatchEvent(new Event('blur')))
    expect(raf.pending()).toBe(1)

    act(() => window.dispatchEvent(new Event('focus')))
    expect(raf.pending()).toBe(1)

    act(() => raf.runNext(0))
    expect(vi.getTimerCount()).toBe(1)

    mount.unmount()
    expect(raf.pending()).toBe(0)
    expect(vi.getTimerCount()).toBe(0)

    act(() => {
      vi.advanceTimersByTime(500)
      window.dispatchEvent(new Event('focus'))
    })
    expect(raf.pending()).toBe(0)
  })
})

describe('roamHopRow', () => {
  it("uses the sheet's own jump row for roam hops, never a remapped state", () => {
    expect(roamHopRow('jump', ['idle', 'jumping', 'smiling'])).toBe('jumping')
    expect(roamHopRow('jump')).toBe('jumping')
    expect(roamHopRow('walk', ['idle', 'jumping'])).toBeUndefined()
    expect(roamHopRow('jump', ['idle'])).toBeUndefined()
  })
})
