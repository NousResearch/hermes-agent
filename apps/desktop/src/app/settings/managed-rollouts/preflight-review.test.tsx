import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { PreflightReview } from './preflight-review'

const draft = { mode: 'manual' as const, concurrency: 1, canaryInstallId: 'b', selectedInstallIds: ['a', 'b'] }

describe('managed rollout preflight', () => {
  it('blocks incompatible starts and suppresses duplicate Start', () => {
    const onStart = vi.fn()
    const planner = () => [['b'], ['a']]
    const { rerender } = render(<PreflightReview compatible={false} draft={draft} onStart={onStart} planner={planner} reviewedToken="t1" />)
    expect(screen.getByRole('button', { name: /start rollout/i })).toHaveProperty('disabled', true)
    rerender(<PreflightReview compatible draft={draft} onStart={onStart} planner={planner} reviewedToken="t1" />)
    fireEvent.click(screen.getByRole('checkbox'))
    const start = screen.getByRole('button', { name: /start rollout/i })
    fireEvent.click(start)
    fireEvent.click(start)
    expect(onStart).toHaveBeenCalledTimes(1)
  })
})
