import { afterEach, describe, expect, it, vi } from 'vitest'

import { observeActiveTerminalResize } from './active-resize'

afterEach(() => {
  vi.unstubAllGlobals()
})

function installRaf() {
  let nextId = 1
  const frames = new Map<number, FrameRequestCallback>()

  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    const id = nextId++
    frames.set(id, callback)

    return id
  })
  vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))

  return {
    flush() {
      const pending = [...frames.entries()]
      frames.clear()
      pending.forEach(([, callback]) => callback(0))
    },
    pending: () => frames.size
  }
}

describe('observeActiveTerminalResize', () => {
  it('fits once on activation and coalesces later resize bursts', () => {
    const raf = installRaf()
    const resize = { current: null as ResizeObserverCallback | null }
    const disconnect = vi.fn()

    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(callback: ResizeObserverCallback) {
          resize.current = callback
        }

        disconnect = disconnect
        observe = vi.fn((target: Element) => {
          resize.current?.([{ target } as ResizeObserverEntry], this as unknown as ResizeObserver)
        })
        unobserve = vi.fn()
      } as unknown as typeof ResizeObserver
    )

    const onFit = vi.fn()
    const onActivate = vi.fn()
    const host = document.createElement('div')
    const dispose = observeActiveTerminalResize(host, { onActivate, onFit })

    // ResizeObserver's initial delivery is absorbed by the activation frame.
    expect(raf.pending()).toBe(1)
    raf.flush()
    expect(onFit).toHaveBeenCalledTimes(1)
    expect(onActivate).toHaveBeenCalledTimes(1)

    resize.current?.([], {} as ResizeObserver)
    resize.current?.([], {} as ResizeObserver)
    resize.current?.([], {} as ResizeObserver)
    expect(raf.pending()).toBe(1)
    raf.flush()
    expect(onFit).toHaveBeenCalledTimes(2)

    dispose()
    expect(disconnect).toHaveBeenCalledTimes(1)
  })

  it('cancels activation without fitting when hidden before the first frame', () => {
    const raf = installRaf()

    vi.stubGlobal(
      'ResizeObserver',
      class {
        disconnect = vi.fn()
        observe = vi.fn()
        unobserve = vi.fn()
      } as unknown as typeof ResizeObserver
    )

    const onFit = vi.fn()

    const dispose = observeActiveTerminalResize(document.createElement('div'), {
      onActivate: vi.fn(),
      onFit
    })

    dispose()

    raf.flush()
    expect(onFit).not.toHaveBeenCalled()
  })

  it('absorbs a real browser-style initial resize delivered after activation', () => {
    const raf = installRaf()
    const resize = { current: null as ResizeObserverCallback | null }

    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(callback: ResizeObserverCallback) {
          resize.current = callback
        }

        disconnect = vi.fn()
        observe = vi.fn()
        unobserve = vi.fn()
      } as unknown as typeof ResizeObserver
    )

    const onFit = vi.fn()
    observeActiveTerminalResize(document.createElement('div'), { onActivate: vi.fn(), onFit })

    raf.flush()
    expect(onFit).toHaveBeenCalledTimes(1)

    // Browser initial delivery: the activation fit already covered this size.
    resize.current?.([], {} as ResizeObserver)
    expect(raf.pending()).toBe(0)

    // A later real resize schedules exactly one fit.
    resize.current?.([], {} as ResizeObserver)
    expect(raf.pending()).toBe(1)
    raf.flush()
    expect(onFit).toHaveBeenCalledTimes(2)
  })

  it('reuses a first-mount fit without fitting again on activation', () => {
    const raf = installRaf()

    vi.stubGlobal(
      'ResizeObserver',
      class {
        disconnect = vi.fn()
        observe = vi.fn()
        unobserve = vi.fn()
      } as unknown as typeof ResizeObserver
    )

    const onActivate = vi.fn()
    const onFit = vi.fn()

    observeActiveTerminalResize(document.createElement('div'), {
      fitOnActivate: false,
      onActivate,
      onFit
    })

    raf.flush()

    expect(onActivate).toHaveBeenCalledOnce()
    expect(onFit).not.toHaveBeenCalled()
  })
})
