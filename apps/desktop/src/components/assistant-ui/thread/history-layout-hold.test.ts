import { afterEach, describe, expect, it, vi } from 'vitest'

import { holdHistoryScroll } from './history-scroll'

const rect = (top: number, height: number): DOMRect =>
  ({ top, bottom: top + height, height, width: 800 } as DOMRect)

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('history layout hold', () => {
  it.each(['resize', 'mutation'] as const)('keeps the anchor through a late %s and releases all pending work', cause => {
    const viewport = window.document.createElement('div')
    const content = window.document.createElement('div')
    const part = window.document.createElement('div')
    part.dataset.historyAnchor = 'text-99'
    content.append(part)
    viewport.append(content)
    viewport.scrollTop = 900
    let displacement = 0
    vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 600))
    vi.spyOn(part, 'getBoundingClientRect').mockImplementation(() => rect(1000 + displacement - viewport.scrollTop, 100))

    let resize!: ResizeObserverCallback
    let mutation!: MutationCallback
    const resizeDisconnect = vi.fn()
    const mutationDisconnect = vi.fn()
    vi.stubGlobal('ResizeObserver', class {
      constructor(callback: ResizeObserverCallback) { resize = callback }
      observe() {}
      disconnect = resizeDisconnect
    })
    vi.stubGlobal('MutationObserver', class {
      constructor(callback: MutationCallback) { mutation = callback }
      observe() {}
      disconnect = mutationDisconnect
    })

    let serial = 0
    const frames = new Map<number, FrameRequestCallback>()
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frames.set(++serial, callback)

      return serial
    })
    vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))

    const tick = () => {
      const callbacks = [...frames.values()]
      frames.clear()
      callbacks.forEach(callback => callback(0))
    }

    const stop = vi.fn()
    const release = holdHistoryScroll(viewport, content, [{ key: 'text-99', occurrence: 0, offset: 100 }], stop)
    tick()
    const stopped = stop.mock.calls.length

    // No busy animation loop, even though the hold survives many idle frames.
    for (let frame = 0; frame < 20; frame++) {
      tick()
    }

    expect(stop).toHaveBeenCalledTimes(stopped)
    displacement = -800

    if (cause === 'resize') {
      resize([], {} as ResizeObserver)
    } else {
      mutation([], {} as MutationObserver)
      mutation([], {} as MutationObserver)
      expect(frames.size).toBe(1)
      tick()
    }

    expect(part.getBoundingClientRect().top).toBe(100)
    expect(viewport.scrollTop).toBe(100)
    mutation([], {} as MutationObserver)
    expect(frames.size).toBe(1)
    release()
    expect(frames.size).toBe(0)
    expect(resizeDisconnect).toHaveBeenCalledOnce()
    expect(mutationDisconnect).toHaveBeenCalledOnce()
    displacement = 400
    resize([], {} as ResizeObserver)
    mutation([], {} as MutationObserver)
    tick()
    expect(viewport.scrollTop).toBe(100)
  })
})
