import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { RolloutControls } from './rollout-controls'

afterEach(() => {
  vi.unstubAllGlobals()
})

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

  it('keeps focus and action state deterministic when reduced motion is requested', () => {
    vi.stubGlobal(
      'matchMedia',
      vi.fn().mockReturnValue({
        addEventListener: vi.fn(),
        matches: true,
        media: '(prefers-reduced-motion: reduce)',
        removeEventListener: vi.fn()
      })
    )
    const onCommand = vi.fn()
    const onVerify = vi.fn()
    const onSubmit = vi.fn(event => event.preventDefault())

    render(
      <form onSubmit={onSubmit}>
        <RolloutControls onCommand={onCommand} onVerify={onVerify} phase="running" />
      </form>
    )

    const pause = screen.getByRole('button', { name: 'Pause' })
    expect(pause.getAttribute('type')).toBe('button')
    expect(pause.className).toContain('motion-reduce:transition-none')

    pause.focus()
    fireEvent.click(pause)

    expect(document.activeElement).toBe(pause)
    expect(screen.getByRole('button', { name: 'Pausing…' })).toHaveProperty('disabled', true)
    expect(onCommand).toHaveBeenCalledWith('pause')
    expect(onSubmit).not.toHaveBeenCalled()
  })
})
