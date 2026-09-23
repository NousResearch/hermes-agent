import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { AttemptDetail } from './attempt-detail'

describe('managed rollout attempt detail', () => {
  it('keeps receipt, readiness, unknown outcome, and recovery fence distinct', () => {
    const onToggle = vi.fn()

    const attempt = {
      installId: 'a'.repeat(32),
      phase: 'unverified',
      expanded: true,
      receipt: { outcome: 'unverified', correlationId: 'correlation-a' },
      readiness: { ready: false, reason: 'scope evidence missing' },
      unknown: true,
      fenced: true
    }

    render(<AttemptDetail attempt={attempt} onToggle={onToggle} />)

    expect(screen.getByText(/Receipt: unverified \(correlation-a\)/)).toBeTruthy()
    expect(screen.getByText(/Readiness: scope evidence missing/)).toBeTruthy()
    expect(screen.getByText(/Remote outcome remains unknown/)).toBeTruthy()
    expect(screen.getByText(/Recovery fence remains active/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { expanded: true }))
    expect(onToggle).toHaveBeenCalledWith(attempt.installId)
  })
})
