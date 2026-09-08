import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { installWindowStateBridge, type WindowStateBridge } from '../test/window-state'

import { createBudgetedLoop } from './budgeted-loop'

let windowState: WindowStateBridge

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

describe('createBudgetedLoop', () => {
  beforeEach(() => {
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    windowState = installWindowStateBridge()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('enforces the fps budget: frames inside the interval reschedule without drawing', () => {
    const raf = installRaf()
    const draw = vi.fn()
    const loop = createBudgetedLoop(draw, { fps: 15 })

    expect(raf.pending()).toBe(1)

    raf.runNext(1000)
    expect(draw).toHaveBeenCalledTimes(1)

    raf.runNext(1010) // inside the ~66.7ms budget
    expect(draw).toHaveBeenCalledTimes(1)
    expect(raf.pending()).toBe(1) // loop stays alive

    raf.runNext(1080) // past the budget
    expect(draw).toHaveBeenCalledTimes(2)

    loop.dispose()
  })

  it('pauses while the window is not observable and resumes when it is', () => {
    const raf = installRaf()
    const draw = vi.fn()
    const loop = createBudgetedLoop(draw)

    expect(raf.pending()).toBe(1)

    window.dispatchEvent(new Event('blur'))
    expect(raf.pending()).toBe(0)

    window.dispatchEvent(new Event('focus'))
    expect(raf.pending()).toBe(1)

    windowState.emit({ isMinimized: true, isVisible: false })
    expect(raf.pending()).toBe(0)

    windowState.emit({ isMinimized: false, isVisible: true })
    expect(raf.pending()).toBe(1)

    loop.dispose()
  })

  it('parks when idleWhen reports nothing to animate and wakes on demand', () => {
    const raf = installRaf()
    const draw = vi.fn()
    let idle = false
    const loop = createBudgetedLoop(draw, { fps: 15, idleWhen: () => idle })

    raf.runNext(1000)
    expect(raf.pending()).toBe(1) // busy -> keeps running

    idle = true
    raf.runNext(1100)
    expect(raf.pending()).toBe(0) // parked, zero pending work
    expect(loop.isDormant()).toBe(true)

    loop.wake() // still idle — but wake schedules; next tick re-parks
    expect(raf.pending()).toBe(1)
    raf.runNext(1200)
    expect(raf.pending()).toBe(0)

    idle = false
    loop.wake()
    expect(raf.pending()).toBe(1)
    raf.runNext(1300)
    expect(loop.isDormant()).toBe(false)
    expect(raf.pending()).toBe(1)

    loop.dispose()
  })

  it('a parked loop does not resume from focus/visibility churn alone', () => {
    const raf = installRaf()
    let idle = true
    const loop = createBudgetedLoop(vi.fn(), { idleWhen: () => idle })

    raf.runNext(1000)
    expect(raf.pending()).toBe(0)
    expect(loop.isDormant()).toBe(true)

    window.dispatchEvent(new Event('blur'))
    window.dispatchEvent(new Event('focus'))
    expect(raf.pending()).toBe(0) // dormancy survives observability churn

    loop.dispose()
  })

  it('dispose cancels the pending frame and makes wake a no-op', () => {
    const raf = installRaf()
    const draw = vi.fn()
    const loop = createBudgetedLoop(draw)

    expect(raf.pending()).toBe(1)

    loop.dispose()
    expect(raf.pending()).toBe(0)

    loop.wake()
    window.dispatchEvent(new Event('focus'))
    expect(raf.pending()).toBe(0)

    loop.dispose() // idempotent
  })
})
