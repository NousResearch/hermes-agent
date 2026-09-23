import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { RecoveryPanel } from './recovery-panel'

describe('managed rollout recovery', () => {
  it('keeps Recheck distinct from fenced Recover and exposes exclusion/stop', () => {
    const onRecheck = vi.fn(); const onRecover = vi.fn(); const onRetry = vi.fn(); const onExclude = vi.fn(); const onStop = vi.fn()
    render(<RecoveryPanel onExclude={onExclude} onRecheck={onRecheck} onRecover={onRecover} onRetry={onRetry} onStop={onStop} target={{ installId: 'i1', phase: 'unknown', unknown: true, fenced: true, reason: 'lost receipt' }} />)
    fireEvent.click(screen.getByRole('button', { name: 'Recheck' }))
    fireEvent.click(screen.getByRole('button', { name: 'Recover' }))
    expect(screen.getByRole('button', { name: 'Exclude' })).toHaveProperty('disabled', true)
    fireEvent.change(screen.getByRole('textbox', { name: 'Exclusion reason' }), { target: { value: 'Operator excludes this installation' } })
    fireEvent.click(screen.getByRole('button', { name: 'Exclude' }))
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
    expect(onRecheck).toHaveBeenCalledTimes(1)
    expect(onRecover).toHaveBeenCalledTimes(1)
    expect(onExclude).toHaveBeenCalledWith('Operator excludes this installation')
    expect(onStop).toHaveBeenCalledTimes(1)
  })
})
