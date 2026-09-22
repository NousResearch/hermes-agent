import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { RolloutHistory } from './rollout-history'

describe('managed rollout history', () => {
  it('bounds history and preserves selected detail identity', () => {
    const onSelect = vi.fn(); const entries = Array.from({ length: 51 }, (_, index) => ({ id: `r${index}`, phase: 'completed', updatedAt: 'now', unresolved: 0, archived: false, reason: null }))
    render(<RolloutHistory entries={entries} onSelect={onSelect} />)
    expect(screen.getByText(/r0/)).toBeTruthy(); expect(screen.queryByText(/r50/)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /r0/ })); expect(onSelect).toHaveBeenCalledWith('r0')
  })
})
