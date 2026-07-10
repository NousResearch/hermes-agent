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
    cleanup()
    vi.useRealTimers()
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

  it('resets elapsed time when the key changes (new run)', () => {
    const first = render(<Probe active timerKey="run:msg-1" />)

    act(() => {
      vi.advanceTimersByTime(10_000)
    })

    expect(screen.getByTestId('elapsed').textContent).toBe('10')

    first.unmount()

    // A new prompt produces a new message id -> the timer must restart from 0.
    render(<Probe active timerKey="run:msg-2" />)

    expect(screen.getByTestId('elapsed').textContent).toBe('0')
  })

  it('survives a navigation away/back (remount with the same run key)', () => {
    // Simulates the chat view unmounting (chat switch / Settings) and remounting
    // while the same agent turn is still running. The run: key must persist.
    const first = render(<Probe active timerKey="run:msg-1" />)

    act(() => {
      vi.advanceTimersByTime(12_000)
    })

    expect(screen.getByTestId('elapsed').textContent).toBe('12')

    first.unmount()

    // Time passes while the view is away; the run keeps counting in the registry.
    act(() => {
      vi.advanceTimersByTime(5_000)
    })

    render(<Probe active timerKey="run:msg-1" />)

    expect(screen.getByTestId('elapsed').textContent).toBe('17')
  })
})
