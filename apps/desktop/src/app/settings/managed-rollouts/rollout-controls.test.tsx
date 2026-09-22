import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { RolloutControls } from './rollout-controls'

describe('rollout controls', () => {
  it('keeps pause/stop pending until acknowledgement and exposes fresh verification', () => {
    const onCommand = vi.fn()
    const onVerify = vi.fn()
    const view = render(<RolloutControls onCommand={onCommand} onVerify={onVerify} phase="running" />)
    fireEvent.click(screen.getByRole('button', { name: 'Pause' }))
    expect(screen.getByRole('button', { name: 'Pausing…' })).toHaveProperty('disabled', true)
    view.rerender(<RolloutControls onCommand={onCommand} onVerify={onVerify} phase="paused" />)
    fireEvent.click(screen.getByRole('button', { name: 'Verify before promotion' }))
    view.rerender(<RolloutControls onCommand={onCommand} onVerify={onVerify} phase="running" />)
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
    expect(screen.getByRole('button', { name: 'Stopping…' })).toHaveProperty('disabled', true)
    expect(onCommand).toHaveBeenNthCalledWith(1, 'pause')
    expect(onCommand).toHaveBeenNthCalledWith(2, 'stop')
    expect(onVerify).toHaveBeenCalledTimes(1)
  })
})
