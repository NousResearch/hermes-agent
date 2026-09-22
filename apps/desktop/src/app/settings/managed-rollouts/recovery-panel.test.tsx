import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { RecoveryPanel } from './recovery-panel'

describe('managed rollout recovery', () => {
  it('keeps Recheck distinct from fenced Recover and exposes exclusion/stop', () => {
    const onRecheck = vi.fn(); const onRecover = vi.fn(); const onRetry = vi.fn(); const onExclude = vi.fn(); const onStop = vi.fn()
    render(<RecoveryPanel target={{ installId: 'i1', phase: 'unknown', unknown: true, fenced: true, reason: 'lost receipt' }} onRecheck={onRecheck} onRecover={onRecover} onRetry={onRetry} onExclude={onExclude} onStop={onStop} />)
    fireEvent.click(screen.getByRole('button', { name: 'Recheck' })); fireEvent.click(screen.getByRole('button', { name: 'Recover' })); fireEvent.click(screen.getByRole('button', { name: 'Exclude' })); fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
    expect(onRecheck).toHaveBeenCalledTimes(1); expect(onRecover).toHaveBeenCalledTimes(1); expect(onExclude).toHaveBeenCalledTimes(1); expect(onStop).toHaveBeenCalledTimes(1)
  })
})
