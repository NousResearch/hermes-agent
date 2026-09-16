import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { prewarmProfileBackend } = vi.hoisted(() => ({
  prewarmProfileBackend: vi.fn()
}))

vi.mock('@/store/profile', () => ({ prewarmProfileBackend }))

import { useProfilePrewarm } from './use-profile-prewarm'

const DWELL_MS = 120

describe('useProfilePrewarm (#100548 layout-only pointerenter)', () => {
  beforeEach(() => {
    prewarmProfileBackend.mockClear()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('does not spawn on pointerenter without a real pointer move (layout shift under a stationary cursor)', () => {
    const { result } = renderHook(() => useProfilePrewarm('scout'))

    act(() => {
      result.current.startPrewarm()
    })
    act(() => {
      vi.advanceTimersByTime(DWELL_MS)
    })

    expect(prewarmProfileBackend).not.toHaveBeenCalled()
  })

  it('does not cascade when successive rows (distinct profiles) receive layout-only pointerenter', () => {
    const first = renderHook(() => useProfilePrewarm('finance'))
    act(() => {
      first.result.current.startPrewarm()
    })
    act(() => {
      vi.advanceTimersByTime(DWELL_MS)
    })
    first.unmount()

    const second = renderHook(() => useProfilePrewarm('scout'))
    act(() => {
      second.result.current.startPrewarm()
    })
    act(() => {
      vi.advanceTimersByTime(DWELL_MS)
    })

    expect(prewarmProfileBackend).not.toHaveBeenCalled()
  })

  it('still prewarms after enter + pointer move + dwell (genuine hover intent)', () => {
    const { result } = renderHook(() => useProfilePrewarm('scout'))

    act(() => {
      result.current.startPrewarm()
      result.current.notePointerMove()
    })
    act(() => {
      vi.advanceTimersByTime(DWELL_MS)
    })

    expect(prewarmProfileBackend).toHaveBeenCalledTimes(1)
    expect(prewarmProfileBackend).toHaveBeenCalledWith('scout')
  })

  it('cancels an in-flight dwell on pointerleave', () => {
    const { result } = renderHook(() => useProfilePrewarm('scout'))

    act(() => {
      result.current.startPrewarm()
      result.current.notePointerMove()
    })
    act(() => {
      result.current.cancelPrewarm()
    })
    act(() => {
      vi.advanceTimersByTime(DWELL_MS)
    })

    expect(prewarmProfileBackend).not.toHaveBeenCalled()
  })

  it('ignores pointermove that was not preceded by pointerenter on this visit', () => {
    const { result } = renderHook(() => useProfilePrewarm('scout'))

    act(() => {
      result.current.notePointerMove()
    })
    act(() => {
      vi.advanceTimersByTime(DWELL_MS)
    })

    expect(prewarmProfileBackend).not.toHaveBeenCalled()
  })
})
