import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { RolloutControls } from './rollout-controls'

describe('rollout controls', () => {
  it('keeps pause/stop pending until acknowledgement and exposes fresh verification', () => {
    const onCommand = vi.fn()
    const onVerify = vi.fn()
    render(<RolloutControls onCommand={onCommand} onVerify={onVerify} phase="running" />)
    fireEvent.click(screen.getByRole('button', { name: 'Pause' }))
    fireEvent.click(screen.getByRole('button', { name: 'Stopping…' }))
    fireEvent.click(screen.getByRole('button', { name: 'Verify before promotion' }))
    expect(onCommand).toHaveBeenCalledWith('pause')
    expect(onVerify).toHaveBeenCalledTimes(1)
  })
})
