// Two-phase render budget (the session-SWITCH perf lever). Contract:
//  1. Starts at the slim SWITCH_RENDER_BUDGET so the switch commit is cheap.
//  2. Idle-time steps raise it by RAISE_STEP until RENDER_BUDGET.
//  3. onBeforeRaise runs before every step commit (scroll-restore hook).
//  4. A sessionKey change resets to the slim budget and restarts the raise.
//  5. The raise never lowers a budget "Show earlier" already grew.
import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  RAISE_STEP,
  RENDER_BUDGET,
  SWITCH_RENDER_BUDGET,
  useTwoPhaseRenderBudget
} from './use-two-phase-render-budget'

type IdleWindow = Window & {
  cancelIdleCallback?: (handle: number) => void
  requestIdleCallback?: (callback: () => void, opts?: { timeout: number }) => number
}

// jsdom's Window types requestIdleCallback as non-optional — write through an
// untyped record view to stub/remove it without fighting the lib types.
const idleWindow = window as IdleWindow

const setIdleApi = (
  request: ((callback: () => void, opts?: { timeout: number }) => number) | undefined,
  cancel: ((handle: number) => void) | undefined
) => {
  Reflect.set(window, 'requestIdleCallback', request)
  Reflect.set(window, 'cancelIdleCallback', cancel)
}

describe('useTwoPhaseRenderBudget', () => {
  let idleCallbacks: Array<() => void>
  let cancelled: number[]
  let originalRequestIdle: IdleWindow['requestIdleCallback']
  let originalCancelIdle: IdleWindow['cancelIdleCallback']

  const flushIdle = () => {
    const pending = idleCallbacks
    idleCallbacks = []

    for (const callback of pending) {
      callback()
    }
  }

  beforeEach(() => {
    idleCallbacks = []
    cancelled = []
    originalRequestIdle = idleWindow.requestIdleCallback
    originalCancelIdle = idleWindow.cancelIdleCallback
    setIdleApi(
      (callback: () => void) => {
        idleCallbacks.push(callback)

        return idleCallbacks.length
      },
      (handle: number) => {
        cancelled.push(handle)
      }
    )
  })

  afterEach(() => {
    setIdleApi(originalRequestIdle, originalCancelIdle)
    vi.restoreAllMocks()
  })

  it('starts at the slim switch budget', () => {
    const { result } = renderHook(() => useTwoPhaseRenderBudget('s1', () => {}))

    expect(result.current[0]).toBe(SWITCH_RENDER_BUDGET)
  })

  it('raises stepwise on idle until the full budget', () => {
    const { result } = renderHook(() => useTwoPhaseRenderBudget('s1', () => {}))

    act(flushIdle)
    expect(result.current[0]).toBe(Math.min(SWITCH_RENDER_BUDGET + RAISE_STEP, RENDER_BUDGET))

    // Flush until settled (bounded loop — the hook must terminate).
    for (let i = 0; i < 10 && result.current[0] < RENDER_BUDGET; i++) {
      act(flushIdle)
    }

    expect(result.current[0]).toBe(RENDER_BUDGET)

    // Settled: the final queued step no-ops without rescheduling.
    act(flushIdle)
    expect(result.current[0]).toBe(RENDER_BUDGET)
    expect(idleCallbacks.length).toBe(0)
  })

  it('calls onBeforeRaise before each step', () => {
    const onBeforeRaise = vi.fn()
    const { result } = renderHook(() => useTwoPhaseRenderBudget('s1', onBeforeRaise))

    act(flushIdle)
    expect(onBeforeRaise).toHaveBeenCalledTimes(1)

    for (let i = 0; i < 10 && result.current[0] < RENDER_BUDGET; i++) {
      act(flushIdle)
    }

    expect(onBeforeRaise.mock.calls.length).toBeGreaterThanOrEqual(2)
  })

  it('resets to the slim budget when the session switches, then raises again', () => {
    const { rerender, result } = renderHook(({ key }) => useTwoPhaseRenderBudget(key, () => {}), {
      initialProps: { key: 's1' }
    })

    for (let i = 0; i < 10 && result.current[0] < RENDER_BUDGET; i++) {
      act(flushIdle)
    }

    expect(result.current[0]).toBe(RENDER_BUDGET)

    rerender({ key: 's2' })
    expect(result.current[0]).toBe(SWITCH_RENDER_BUDGET)

    for (let i = 0; i < 10 && result.current[0] < RENDER_BUDGET; i++) {
      act(flushIdle)
    }

    expect(result.current[0]).toBe(RENDER_BUDGET)
  })

  it('never lowers a budget the user grew via Show earlier', () => {
    const { result } = renderHook(() => useTwoPhaseRenderBudget('s1', () => {}))

    // Simulate "Show earlier": grow past the steady-state budget.
    act(() => {
      result.current[1](() => RENDER_BUDGET * 2)
    })
    expect(result.current[0]).toBe(RENDER_BUDGET * 2)

    act(flushIdle)
    expect(result.current[0]).toBe(RENDER_BUDGET * 2)
  })

  it('falls back to setTimeout when requestIdleCallback is unavailable', () => {
    vi.useFakeTimers()
    setIdleApi(undefined, undefined)

    try {
      const { result } = renderHook(() => useTwoPhaseRenderBudget('s1', () => {}))

      expect(result.current[0]).toBe(SWITCH_RENDER_BUDGET)

      for (let i = 0; i < 10 && result.current[0] < RENDER_BUDGET; i++) {
        act(() => {
          vi.runOnlyPendingTimers()
        })
      }

      expect(result.current[0]).toBe(RENDER_BUDGET)
    } finally {
      vi.useRealTimers()
    }
  })

  it('cancels the pending idle raise on unmount', () => {
    const { unmount } = renderHook(() => useTwoPhaseRenderBudget('s1', () => {}))

    unmount()
    expect(cancelled.length).toBeGreaterThan(0)
  })
})
