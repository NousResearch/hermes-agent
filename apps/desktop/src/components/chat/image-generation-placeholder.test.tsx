import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useResizeObserver } from '@/hooks/use-resize-observer'
import { setReducedEffects } from '@/store/reduced-effects'

import { installWindowStateBridge, type WindowStateBridge } from '../../test/window-state'

import { DiffusionCanvas } from './image-generation-placeholder'

vi.mock('@/hooks/use-resize-observer', () => ({ useResizeObserver: vi.fn() }))

let root: Root | null = null
let container: HTMLDivElement | null = null
let windowState: WindowStateBridge
let fillRect: ReturnType<typeof vi.fn>

function render(count = 1) {
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)

  act(() => {
    root!.render(
      <>
        {Array.from({ length: count }, (_, index) => (
          <DiffusionCanvas key={index} />
        ))}
      </>
    )
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

describe('DiffusionCanvas scheduling', () => {
  beforeEach(() => {
    setReducedEffects(false)
    ;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    windowState = installWindowStateBridge()
    fillRect = vi.fn()
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      clearRect: vi.fn(),
      createLinearGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
      createRadialGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
      fillRect,
      fillText: vi.fn(),
      setTransform: vi.fn()
    } as unknown as CanvasRenderingContext2D)
  })

  afterEach(() => {
    cleanup()
    setReducedEffects(false)
    vi.mocked(useResizeObserver).mockClear()
    vi.restoreAllMocks()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('keeps animating while unfocused but cancels its loop while minimized', () => {
    const raf = installRaf()

    render()
    expect(raf.request).toHaveBeenCalledTimes(1)
    expect(raf.pending()).toBe(1)

    act(() => {
      window.dispatchEvent(new Event('blur'))
    })
    expect(raf.pending()).toBe(1)

    act(() => {
      window.dispatchEvent(new Event('focus'))
    })
    expect(raf.pending()).toBe(1)

    act(() => {
      windowState.emit({ isMinimized: true, isVisible: false })
    })
    expect(raf.pending()).toBe(0)

    cleanup()
    act(() => {
      window.dispatchEvent(new Event('focus'))
    })
    expect(raf.pending()).toBe(0)
  })

  it('paints a static placeholder without scheduling frames in Reduced mode', () => {
    setReducedEffects(true)
    const raf = installRaf()
    render()
    expect(fillRect).toHaveBeenCalled()
    expect(raf.pending()).toBe(0)
    expect(raf.request).not.toHaveBeenCalled()
  })

  it('disposes its active loop on Reduced and restarts when Full is restored', () => {
    const raf = installRaf()
    render()
    expect(raf.pending()).toBe(1)

    act(() => setReducedEffects(true))
    expect(raf.pending()).toBe(0)
    expect(fillRect).toHaveBeenCalled()

    act(() => setReducedEffects(false))
    expect(raf.pending()).toBe(1)
  })

  it('repaints static content after resizing or a theme change without a loop', async () => {
    setReducedEffects(true)
    const raf = installRaf()
    render()
    fillRect.mockClear()

    act(() => vi.mocked(useResizeObserver).mock.calls.at(-1)![0]([]))
    expect(fillRect).toHaveBeenCalled()
    fillRect.mockClear()

    await act(async () => window.document.documentElement.style.setProperty('--dt-primary', '#123456'))
    expect(fillRect).toHaveBeenCalled()
    expect(raf.pending()).toBe(0)
    window.document.documentElement.style.removeProperty('--dt-primary')
  })

  it('keeps capped Full-mode instances static and visible after resize and theme changes', async () => {
    const raf = installRaf()
    render(3)
    expect(raf.pending()).toBe(2)
    expect(fillRect).toHaveBeenCalled()
    fillRect.mockClear()

    act(() => vi.mocked(useResizeObserver).mock.calls.at(-1)![0]([]))
    expect(fillRect).toHaveBeenCalled()
    fillRect.mockClear()

    await act(async () => window.document.documentElement.style.setProperty('--dt-primary', '#654321'))
    expect(fillRect).toHaveBeenCalled()
    expect(raf.pending()).toBe(2)
    expect(raf.request).toHaveBeenCalledTimes(2)
    window.document.documentElement.style.removeProperty('--dt-primary')
  })
})
