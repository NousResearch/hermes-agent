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

  it('renews an unchanged token without reconfirmation and starts against the renewed token', () => {
    const onStart = vi.fn()
    const onRenew = vi.fn()
    const planner = () => [['b'], ['a']]
    const { rerender } = render(<PreflightReview compatible draft={draft} key="digest-a" onRenew={onRenew} onStart={onStart} planner={planner} reviewedToken="t1" />)

    fireEvent.click(screen.getByRole('checkbox'))
    expect(screen.getByRole('checkbox')).toHaveProperty('checked', true)

    // Same plan digest, new token: the review stays mounted, the operator's
    // confirmation carries over, and Start binds to the renewed token.
    rerender(<PreflightReview compatible draft={draft} key="digest-a" onRenew={onRenew} onStart={onStart} planner={planner} reviewedToken="t2" />)
    fireEvent.click(screen.getByRole('button', { name: /renew review/i }))
    expect(onRenew).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('checkbox')).toHaveProperty('checked', true)
    expect(screen.getByRole('button', { name: /start rollout/i })).toHaveProperty('disabled', false)

    fireEvent.click(screen.getByRole('button', { name: /start rollout/i }))
    expect(onStart).toHaveBeenCalledWith(draft, 't2')
  })

  it('requires reconfirmation after a changed plan row remounts the review', () => {
    const onStart = vi.fn()
    const planner = () => [['b'], ['a']]
    const { rerender } = render(<PreflightReview compatible draft={draft} key="digest-a" onStart={onStart} planner={planner} reviewedToken="t1" />)

    fireEvent.click(screen.getByRole('checkbox'))
    expect(screen.getByRole('checkbox')).toHaveProperty('checked', true)

    // A changed row yields a new plan digest, which remounts the review: the
    // earlier confirmation cannot carry over to the changed plan.
    const changed = { ...draft, selectedInstallIds: ['a', 'b', 'c'] }
    rerender(<PreflightReview compatible draft={changed} key="digest-b" onStart={onStart} planner={() => [['b'], ['a', 'c']]} reviewedToken="t2" />)
    expect(screen.getByRole('checkbox')).toHaveProperty('checked', false)
    expect(screen.getByRole('button', { name: /start rollout/i })).toHaveProperty('disabled', true)
    expect(onStart).not.toHaveBeenCalled()
  })

  it('cannot advance while an incompatible changed row remains unresolved', () => {
    const onStart = vi.fn()
    const planner = () => [['b'], ['a']]
    render(<PreflightReview changes={[{ installId: 'a', field: 'head', before: '1'.repeat(40), after: '2'.repeat(40) }]} compatible draft={draft} onStart={onStart} planner={planner} reviewedToken="t2" />)

    fireEvent.click(screen.getByRole('checkbox'))
    expect(screen.getByRole('button', { name: /start rollout/i })).toHaveProperty('disabled', true)
    fireEvent.click(screen.getByRole('button', { name: /start rollout/i }))
    expect(onStart).not.toHaveBeenCalled()
  })
})
