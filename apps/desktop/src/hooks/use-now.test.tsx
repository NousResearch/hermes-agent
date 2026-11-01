import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { __resetNowClocksForTests, useNow } from './use-now'

function Clock({ interval, label }: { interval: number; label: string }) {
  const now = useNow(interval)

  return <span data-testid={label}>{now}</span>
}

describe('useNow', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-01-01T00:00:00.000Z'))
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    __resetNowClocksForTests()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.useRealTimers()
    __resetNowClocksForTests()
  })

  it('shares one interval across many subscribers at the same cadence', () => {
    const setIntervalSpy = vi.spyOn(window, 'setInterval')
    render(
      <>
        <Clock interval={1000} label="a" />
        <Clock interval={1000} label="b" />
        <Clock interval={1000} label="c" />
      </>
    )

    expect(setIntervalSpy).toHaveBeenCalledTimes(1)

    act(() => {
      vi.advanceTimersByTime(3_000)
    })

    expect(screen.getByTestId('a').textContent).toBe(screen.getByTestId('b').textContent)
    expect(screen.getByTestId('b').textContent).toBe(screen.getByTestId('c').textContent)
  })

  it('ticks each cadence independently and stops the interval when empty', () => {
    const clearSpy = vi.spyOn(window, 'clearInterval')
    const first = render(<Clock interval={1000} label="fast" />)
    render(<Clock interval={5000} label="slow" />)

    act(() => {
      vi.advanceTimersByTime(5_000)
    })

    const fast = Number(screen.getByTestId('fast').textContent)
    const slow = Number(screen.getByTestId('slow').textContent)
    expect(fast).toBeGreaterThan(0)
    expect(slow).toBeGreaterThan(0)

    first.unmount()
    act(() => {
      vi.advanceTimersByTime(10_000)
    })

    // The 1s clock had its only subscriber leave: its interval is cleared,
    // while the 5s clock keeps ticking underneath.
    expect(clearSpy).toHaveBeenCalled()
    expect(Number(screen.getByTestId('slow').textContent)).toBeGreaterThan(slow)
  })

  it('pauses while unfocused and catches up with a leading tick on return', () => {
    render(<Clock interval={1000} label="t" />)
    const atMount = Number(screen.getByTestId('t').textContent)

    vi.mocked(document.hasFocus).mockReturnValue(false)
    window.dispatchEvent(new Event('blur'))
    act(() => {
      vi.advanceTimersByTime(5_000)
    })
    expect(screen.getByTestId('t').textContent).toBe(String(atMount))

    vi.mocked(document.hasFocus).mockReturnValue(true)
    act(() => {
      window.dispatchEvent(new Event('focus'))
    })
    expect(Number(screen.getByTestId('t').textContent)).toBeGreaterThan(atMount)
  })
})
