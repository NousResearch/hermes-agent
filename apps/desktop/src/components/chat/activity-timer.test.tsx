import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { __resetElapsedTimerRegistryForTests, useElapsedSeconds } from './activity-timer'

function Probe({ active, timerKey }: { active: boolean; timerKey?: string }) {
  const elapsed = useElapsedSeconds(active, timerKey)

  return <span data-testid="elapsed">{elapsed}</span>
}

describe('useElapsedSeconds', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-01-01T00:00:00.000Z'))
    __resetElapsedTimerRegistryForTests()
  })

  afterEach(() => {
    vi.useRealTimers()
    cleanup()
    __resetElapsedTimerRegistryForTests()
  })

  it('keeps elapsed time stable across remounts for the same key', () => {
    const first = render(<Probe active timerKey="tool:abc" />)

    act(() => {
      vi.advanceTimersByTime(5_000)
    })

    expect(screen.getByTestId('elapsed').textContent).toBe('5')

    first.unmount()

    act(() => {
      vi.advanceTimersByTime(3_000)
    })

    render(<Probe active timerKey="tool:abc" />)

    expect(screen.getByTestId('elapsed').textContent).toBe('8')
  })

  // Regression: opening a thread-level overlay (settings / agents /
  // command-center) navigates to a different route and unmounts the chat
  // route entirely (see <Routes element={null} path="settings" ...> in
  // desktop-controller). Without a timerKey the elapsed clock resets to 0
  // on every overlay open/close. A per-message timerKey (e.g.
  // stream-stall:{messageId}) keeps the start time in the module registry
  // across the unmount/remount cycle — same message keeps counting,
  // different message resets (the key changes).
  it('simulates overlay remount keeping the clock alive for the same message key', () => {
    const { unmount } = render(<Probe active timerKey="stream-stall:msg-1" />)

    act(() => {
      vi.advanceTimersByTime(10_000)
    })

    expect(screen.getByTestId('elapsed').textContent).toBe('10')

    // User opens settings overlay → chat route unmounts.
    unmount()

    act(() => {
      vi.advanceTimersByTime(7_000)
    })

    // User closes settings overlay → chat route remounts with the same key.
    render(<Probe active timerKey="stream-stall:msg-1" />)

    expect(screen.getByTestId('elapsed').textContent).toBe('17')
  })

  it('resets when the message key changes via rerender', () => {
    const { rerender } = render(<Probe active timerKey="stream-stall:msg-1" />)

    act(() => {
      vi.advanceTimersByTime(10_000)
    })

    expect(screen.getByTestId('elapsed').textContent).toBe('10')

    // Simulate a new turn: the same component instance receives a new
    // key.  The old key should be deleted from the registry.
    rerender(<Probe active timerKey="stream-stall:msg-2" />)

    act(() => {
      vi.advanceTimersByTime(3_000)
    })

    expect(screen.getByTestId('elapsed').textContent).toBe('3')
  })
})
