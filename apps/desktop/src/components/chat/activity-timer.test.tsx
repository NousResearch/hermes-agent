import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { __resetElapsedTimerRegistryForTests, useElapsedSeconds, useMeasuredDuration } from './activity-timer'

function Probe({ active, since, timerKey }: { active: boolean; since?: number; timerKey?: string }) {
  const elapsed = useElapsedSeconds(active, timerKey, since)

  return <span data-testid="elapsed">{elapsed}</span>
}

function DurationProbe({ active, timerKey }: { active: boolean; timerKey: string }) {
  const measured = useMeasuredDuration(active, timerKey)

  return <span data-testid="measured">{measured === null ? 'unknown' : measured}</span>
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

describe('useMeasuredDuration', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-01-01T00:00:00.000Z'))
    __resetElapsedTimerRegistryForTests()
  })

  afterEach(() => {
    vi.useRealTimers()
    __resetElapsedTimerRegistryForTests()
  })

  it('has nothing to report until it has watched something finish', () => {
    render(<DurationProbe active timerKey="reasoning:m1:0" />)

    act(() => {
      vi.advanceTimersByTime(4_000)
    })

    expect(screen.getByTestId('measured').textContent).toBe('unknown')
  })

  it('freezes the duration at the moment the thing finishes', () => {
    const { rerender } = render(<DurationProbe active timerKey="reasoning:m1:0" />)

    act(() => {
      vi.advanceTimersByTime(4_000)
    })

    rerender(<DurationProbe active={false} timerKey="reasoning:m1:0" />)

    expect(screen.getByTestId('measured').textContent).toBe('4')

    // Time keeps passing; the block is over and its duration must not creep up
    // with it.
    act(() => {
      vi.advanceTimersByTime(9_000)
    })

    expect(screen.getByTestId('measured').textContent).toBe('4')
  })

  // The thread virtualizes, so the component that watched a block finish is
  // usually gone by the time anyone scrolls back to read it.
  it('remembers the duration for a component that mounts after the fact', () => {
    const first = render(<DurationProbe active timerKey="reasoning:m1:0" />)

    act(() => {
      vi.advanceTimersByTime(6_000)
    })

    first.rerender(<DurationProbe active={false} timerKey="reasoning:m1:0" />)
    first.unmount()

    render(<DurationProbe active={false} timerKey="reasoning:m1:0" />)

    expect(screen.getByTestId('measured').textContent).toBe('6')
  })

  it('measures each key separately', () => {
    const first = render(<DurationProbe active timerKey="reasoning:m1:0" />)

    act(() => {
      vi.advanceTimersByTime(3_000)
    })

    first.rerender(<DurationProbe active={false} timerKey="reasoning:m1:0" />)
    first.unmount()

    const second = render(<DurationProbe active timerKey="reasoning:m1:7" />)

    act(() => {
      vi.advanceTimersByTime(2_000)
    })

    second.rerender(<DurationProbe active={false} timerKey="reasoning:m1:7" />)

    expect(screen.getByTestId('measured').textContent).toBe('2')
  })
})
