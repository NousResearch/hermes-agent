import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PixelEggSprite } from './pixel-egg-sprite'

let root: Root | null = null
let container: HTMLDivElement | null = null

function render(mode: 'bounce' | 'hatch', onDone?: () => void) {
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)

  act(() => {
    root!.render(<PixelEggSprite mode={mode} onDone={onDone} size={32} />)
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

function installRaf(startId = 1) {
  let nextId = startId
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

describe('PixelEggSprite scheduling', () => {
  beforeEach(() => {
    ;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    vi.useFakeTimers()
    vi.spyOn(Math, 'random').mockReturnValue(0)
    Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: 1 })
    vi.stubGlobal(
      'Image',
      class {
        complete = true
        onerror: (() => undefined) | null = null
        onload: (() => undefined) | null = null

        set src(_value: string) {
          queueMicrotask(() => this.onload?.())
        }
      } as unknown as typeof Image
    )
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      clearRect: vi.fn(),
      drawImage: vi.fn(),
      getImageData: vi.fn(() => ({ data: new Uint8ClampedArray(32 * 32 * 4) })),
      imageSmoothingEnabled: false,
      putImageData: vi.fn()
    } as unknown as CanvasRenderingContext2D)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('sleeps during the long bounce rest instead of chaining RAFs', async () => {
    const raf = installRaf()

    render('bounce')
    await act(async () => undefined)

    act(() => {
      raf.runNext(0)
    })

    expect(raf.pending()).toBe(0)
    expect(vi.getTimerCount()).toBe(1)

    act(() => {
      vi.advanceTimersByTime(650)
    })

    expect(raf.pending()).toBe(1)

    cleanup()
    expect(raf.pending()).toBe(0)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('stops scheduling frames after the final hatch frame', async () => {
    const raf = installRaf()
    const onDone = vi.fn()

    render('hatch', onDone)
    await act(async () => undefined)

    act(() => {
      raf.runNext(0)
    })

    for (const now of [190, 380, 570, 760, 950]) {
      act(() => {
        vi.advanceTimersByTime(162)
      })

      act(() => {
        raf.runNext(now)
      })
    }

    expect(onDone).toHaveBeenCalledTimes(1)
    expect(raf.pending()).toBe(0)
  })

  it('cancels a pending animation frame even when the browser assigns ID zero', async () => {
    const raf = installRaf(0)

    render('bounce')
    await act(async () => undefined)

    expect(raf.pending()).toBe(1)
    cleanup()
    expect(raf.pending()).toBe(0)
  })
})
