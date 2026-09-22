import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { PreparationPanel } from './preparation-panel'

const target = { installId: 'install-a', machineId: 'machine-a', label: 'Lab', supported: true, sharedMachine: false }

describe('managed rollout preparation panel', () => {
  it('requires explicit confirmation and selection before preparation', () => {
    const onPrepare = vi.fn()
    render(<PreparationPanel onPrepare={onPrepare} targets={[target]} />)
    const prepare = screen.getByRole('button', { name: /prepare selected/i })
    expect(prepare).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Select' }))
    expect(prepare).toBeDisabled()
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(prepare)
    expect(onPrepare).toHaveBeenCalledWith([target])
  })

  it('clears confirmation when selection changes, forcing requalification', () => {
    const onPrepare = vi.fn()
    render(<PreparationPanel onPrepare={onPrepare} targets={[target]} />)
    fireEvent.click(screen.getByRole('button', { name: 'Select' }))
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(screen.getByRole('button', { name: 'Selected' }))
    expect(screen.getByRole('button', { name: /prepare selected/i })).toBeDisabled()
  })
})
