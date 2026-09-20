import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { reactRoot } from '@/test/react-root'
import { setDocumentHidden } from '@/test/window-state'

import { Loader } from './loader'

const PAUSED_ATTRIBUTE = 'data-renderer-animations-paused'

/**
 * The paused-time accounting in `Loader` is only meaningful against a clock the
 * test owns: Chromium throttles — and for a hidden document stops — rAF, so the
 * frame timestamps the loader receives are not evenly spaced. This harness hands
 * it exactly the timestamps we want, including a 60 s gap with no callbacks at
 * all, which is what a hidden tab actually looks like.
 */
function installFrameClock() {
  let clock = 1_000
  let nextId = 1
  const frames = new Map<number, FrameRequestCallback>()

  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    const id = nextId++

    frames.set(id, callback)

    return id
  })
  vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))
  vi.spyOn(performance, 'now').mockImplementation(() => clock)

  return {
    /** Advance the clock and run the frames pending right now, like one rAF
     *  turn: a callback that re-requests a frame lands in the next turn. */
    frame(now: number) {
      clock = now

      const pending = [...frames.values()]

      frames.clear()

      pending.forEach(callback => act(() => callback(now)))
    },
    /** Move the clock without running rAF — what a hidden document does. */
    idleUntil(now: number) {
      clock = now
    }
  }
}

const mount = reactRoot()
const path = () => mount.container!.querySelector('path')!.getAttribute('d')

/** jsdom has no layout, so `getBoundingClientRect` is 0×0 everywhere — which
 *  would trip the loader's zero-size pause gate and freeze every frame. */
const rect = { bottom: 40, height: 40, left: 0, right: 40, top: 0, width: 40, x: 0, y: 0 } as DOMRect

function hide() {
  setDocumentHidden(true)
  document.dispatchEvent(new Event('visibilitychange'))
}

function show() {
  setDocumentHidden(false)
  document.dispatchEvent(new Event('visibilitychange'))
}

beforeEach(() => {
  // Fixed phase offset, so two mounts of the same curve are comparable.
  vi.spyOn(Math, 'random').mockReturnValue(0.5)
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(rect)
})

afterEach(() => {
  mount.unmount()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  setDocumentHidden(false)
  document.documentElement.removeAttribute(PAUSED_ATTRIBUTE)
})

describe('Loader paused-time accounting', () => {
  it('freezes progress across a hidden interval where rAF never ran', () => {
    const clock = installFrameClock()

    mount.render(<Loader />)
    clock.frame(1_016)
    const beforeHide = path()

    hide()
    // A hidden document gets no rAF callbacks at all: the clock runs, the
    // animation does not.
    clock.idleUntil(61_000)
    show()
    clock.frame(61_000)

    expect(path()).toBe(beforeHide)
  })

  it('freezes progress when rAF is throttled rather than stopped while hidden', () => {
    const clock = installFrameClock()

    mount.render(<Loader />)
    clock.frame(1_016)
    const beforeHide = path()

    hide()
    // Chromium's throttled case: a callback every ~20 s over the same minute.
    clock.frame(20_000)
    clock.frame(40_000)
    clock.frame(60_000)
    show()
    clock.frame(61_016)

    expect(path()).toBe(beforeHide)
  })

  it('measures a renderer-paused stretch by the clock, not 16 ms per callback', () => {
    const clock = installFrameClock()

    // Reference: the curve as it looks at animation time 0.
    mount.render(<Loader />)
    const neverPaused = path()

    mount.unmount()
    document.documentElement.setAttribute(PAUSED_ATTRIBUTE, '')

    // 2.5 s of paused frames at a 25 ms cadence — a 120 Hz panel or a loaded
    // main thread deliver 16 ms frames no more often than any other cadence.
    mount.render(<Loader />)

    for (let now = 1_025; now <= 3_500; now += 25) {
      clock.frame(now)
    }

    // Paused frames still write nothing: the DOM keeps the curve from the last
    // frame that ran, so the mount frame left no path at all.
    expect(path()).toBeNull()

    document.documentElement.removeAttribute(PAUSED_ATTRIBUTE)
    clock.frame(3_525)

    expect(path()).toBe(neverPaused)
  })
})
