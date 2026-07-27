import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { __resetElapsedTimerRegistryForTests, useElapsedSeconds } from './activity-timer'

function Probe({ active, since, timerKey }: { active: boolean; since?: number; timerKey?: string }) {
  const elapsed = useElapsedSeconds(active, timerKey, since)

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

  it('counts from an explicit epoch rather than mount time', () => {
    const mountedAt = Date.now()

    act(() => {
      vi.advanceTimersByTime(30_000)
    })

    render(<Probe active since={mountedAt + 28_000} />)

    expect(screen.getByTestId('elapsed').textContent).toBe('2')
  })

  it('re-anchors when the epoch moves', () => {
    const { rerender } = render(<Probe active since={Date.now()} />)

    act(() => {
      vi.advanceTimersByTime(10_000)
    })

    expect(screen.getByTestId('elapsed').textContent).toBe('10')

    rerender(<Probe active since={Date.now()} />)

    expect(screen.getByTestId('elapsed').textContent).toBe('0')
  })
})
