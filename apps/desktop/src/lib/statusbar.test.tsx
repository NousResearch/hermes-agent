import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { formatDuration, LiveDuration } from './statusbar'

describe('formatDuration', () => {
  it('formats sub-hour durations as m:ss', () => {
    expect(formatDuration(0)).toBe('0:00')
    expect(formatDuration(59_000)).toBe('0:59')
    expect(formatDuration(61_000)).toBe('1:01')
  })

  it('formats hour-plus durations as h:mm:ss', () => {
    expect(formatDuration(3_600_000)).toBe('1:00:00')
    expect(formatDuration(3_661_000)).toBe('1:01:01')
  })

  it('clamps negative elapsed to zero', () => {
    expect(formatDuration(-5_000)).toBe('0:00')
  })
})

describe('LiveDuration', () => {
  it('renders nothing without a start timestamp', () => {
    const { container } = render(<LiveDuration since={null} />)

    expect(container.innerHTML).toBe('')
  })

  it('renders the elapsed time with fixed-width digits so neighbors do not shift', () => {
    vi.useFakeTimers()

    try {
      const now = Date.now()
      render(<LiveDuration since={now - 71_000} />)

      const el = screen.getByText('1:11')
      // tabular-nums keeps every digit the same width; without it the
      // statusbar row to the left of the session timer jitters each second.
      expect(el.classList.contains('tabular-nums')).toBe(true)
    } finally {
      vi.useRealTimers()
    }
  })
})
