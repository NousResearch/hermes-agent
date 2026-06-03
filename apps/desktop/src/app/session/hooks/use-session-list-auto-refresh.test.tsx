import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useSessionListAutoRefresh } from './use-session-list-auto-refresh'

describe('useSessionListAutoRefresh', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('polls the session list silently while enabled', () => {
    const refreshSessions = vi.fn(async () => undefined)

    renderHook(() =>
      useSessionListAutoRefresh({
        enabled: true,
        intervalMs: 1000,
        refreshSessions
      })
    )

    expect(refreshSessions).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(1000)
    })

    expect(refreshSessions).toHaveBeenCalledTimes(1)
    expect(refreshSessions).toHaveBeenCalledWith({ showLoading: false })
  })

  it('does not poll while disabled', () => {
    const refreshSessions = vi.fn(async () => undefined)

    renderHook(() =>
      useSessionListAutoRefresh({
        enabled: false,
        intervalMs: 1000,
        refreshSessions
      })
    )

    act(() => {
      vi.advanceTimersByTime(5000)
    })

    expect(refreshSessions).not.toHaveBeenCalled()
  })

  it('stops polling after unmount', () => {
    const refreshSessions = vi.fn(async () => undefined)

    const { unmount } = renderHook(() =>
      useSessionListAutoRefresh({
        enabled: true,
        intervalMs: 1000,
        refreshSessions
      })
    )

    unmount()

    act(() => {
      vi.advanceTimersByTime(5000)
    })

    expect(refreshSessions).not.toHaveBeenCalled()
  })
})
