import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PersistentTerminal, TerminalSlot } from './persistent'

vi.mock('./terminals', () => ({ ensureTerminal: vi.fn() }))
vi.mock('./workspace', () => ({ TerminalWorkspace: () => <div data-testid="terminal-workspace" /> }))

const resizeObservers = new Set<TestResizeObserver>()

class TestResizeObserver {
  constructor(private readonly callback: ResizeObserverCallback) {
    resizeObservers.add(this)
  }

  observe() {}

  unobserve() {}

  disconnect() {
    resizeObservers.delete(this)
  }

  notify(target: Element) {
    this.callback([{ target } as ResizeObserverEntry], this as unknown as ResizeObserver)
  }
}

const rect = (left: number, top: number, width: number, height: number): DOMRect =>
  ({
    bottom: top + height,
    height,
    left,
    right: left + width,
    top,
    width,
    x: left,
    y: top,
    toJSON: () => ({})
  }) as DOMRect

describe('PersistentTerminal geometry tracking', () => {
  const frames = new Map<number, FrameRequestCallback>()
  let nextFrame = 1

  beforeEach(() => {
    frames.clear()
    resizeObservers.clear()
    nextFrame = 1

    vi.stubGlobal('ResizeObserver', TestResizeObserver)
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      const id = nextFrame++

      frames.set(id, callback)

      return id
    })
    vi.stubGlobal('cancelAnimationFrame', (id: number) => {
      frames.delete(id)
    })
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(rect(20, 30, 640, 360))
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('stays dormant while geometry is unchanged and coalesces observer updates to one frame', () => {
    const { container, unmount } = render(
      <>
        <TerminalSlot />
        <PersistentTerminal onAddSelectionToChat={() => {}} />
      </>
    )

    const slot = container.firstElementChild!
    const overlay = container.lastElementChild as HTMLElement

    expect(overlay.style.left).toBe('20px')
    expect(overlay.style.top).toBe('30px')
    expect(frames.size).toBe(0)
    expect(resizeObservers.size).toBe(1)

    vi.spyOn(slot, 'getBoundingClientRect').mockReturnValue(rect(40, 50, 600, 320))

    act(() => {
      for (const observer of resizeObservers) {
        observer.notify(slot)
        observer.notify(slot)
      }
    })

    expect(frames.size).toBe(1)

    act(() => {
      const [id, callback] = [...frames.entries()][0]!

      frames.delete(id)
      callback(performance.now())
    })

    expect(overlay.style.left).toBe('40px')
    expect(overlay.style.top).toBe('50px')
    expect(frames.size).toBe(0)

    act(() => {
      for (const observer of resizeObservers) {
        observer.notify(slot)
      }
    })

    expect(frames.size).toBe(1)

    unmount()

    expect(frames.size).toBe(0)
    expect(resizeObservers.size).toBe(0)
  })
})
