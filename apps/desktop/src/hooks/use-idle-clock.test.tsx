// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { IDLE_TICK_MS, useIdleClock } from './use-idle-clock'

let visibility: DocumentVisibilityState = 'visible'
let focused = true

function setViewed(viewed: boolean): void {
  visibility = viewed ? 'visible' : 'hidden'
  focused = viewed
  window.document.dispatchEvent(new Event('visibilitychange'))
  window.dispatchEvent(new Event(viewed ? 'focus' : 'blur'))
}

describe('useIdleClock', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(window.document, 'visibilityState', 'get').mockImplementation(() => visibility)
    vi.spyOn(window.document, 'hasFocus').mockImplementation(() => focused)
    visibility = 'visible'
    focused = true
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  // #122413: the sidebar cron countdowns and the heartbeat countdown each ran
  // their own always-on 1s setInterval, keeping the visible-idle renderer hot.
  it('ticks about once a second while the window is viewed', () => {
    const { result } = renderHook(() => useIdleClock(true))
    const start = result.current

    act(() => {
      vi.advanceTimersByTime(IDLE_TICK_MS * 3 + 50)
    })

    expect(result.current).toBeGreaterThan(start)
  })

  it('parks the tick while hidden and catches up on return', () => {
    const { result } = renderHook(() => useIdleClock(true))
    const setIntervalSpy = vi.spyOn(window, 'setInterval')

    act(() => {
      setViewed(false)
    })
    const parked = result.current
    setIntervalSpy.mockClear()

    act(() => {
      vi.advanceTimersByTime(IDLE_TICK_MS * 5)
    })

    expect(result.current).toBe(parked)
    expect(setIntervalSpy).not.toHaveBeenCalled()

    act(() => {
      setViewed(true)
    })

    // Leading tick on return — the UI catches up immediately, no stale countdown.
    expect(result.current).toBeGreaterThanOrEqual(parked)
  })

  it('starts no timer while disabled', () => {
    const setIntervalSpy = vi.spyOn(window, 'setInterval')
    const { result } = renderHook(() => useIdleClock(false))

    act(() => {
      vi.advanceTimersByTime(IDLE_TICK_MS * 3)
    })

    expect(setIntervalSpy).not.toHaveBeenCalled()
    expect(typeof result.current).toBe('number')
  })

  it('shares one underlying interval across mounted consumers', () => {
    const setIntervalSpy = vi.spyOn(window, 'setInterval')
    const first = renderHook(() => useIdleClock(true))
    const second = renderHook(() => useIdleClock(true))

    expect(setIntervalSpy).toHaveBeenCalledTimes(1)

    act(() => {
      vi.advanceTimersByTime(IDLE_TICK_MS + 50)
    })

    // One shared tick drives every consumer — no per-clock timer multiplication.
    expect(second.result.current).toBe(first.result.current)

    first.unmount()
    second.unmount()
  })

  it('clears the shared interval once the last consumer leaves', () => {
    const clearIntervalSpy = vi.spyOn(window, 'clearInterval')
    const { unmount } = renderHook(() => useIdleClock(true))

    unmount()

    expect(clearIntervalSpy).toHaveBeenCalled()
  })
})
