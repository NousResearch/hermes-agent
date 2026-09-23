import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { RolloutControls } from './rollout-controls'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('rollout controls', () => {
  it('releases pending after accepted command completion even if snapshot refresh has not repainted', async () => {
    const onCommand = vi.fn().mockResolvedValue(true)
    render(<RolloutControls onCommand={onCommand} onVerify={vi.fn()} phase="paused" />)

    fireEvent.click(screen.getByRole('button', { name: 'Resume' }))
    expect(screen.getByRole('button', { name: 'Resume' })).toHaveProperty('disabled', true)

    await act(async () => {await Promise.resolve()})

    expect(screen.getByRole('button', { name: 'Resume' })).toHaveProperty('disabled', false)
  })

  it('offers Resume for a paused rollout and waits for the running acknowledgement', () => {
    const onCommand = vi.fn().mockResolvedValue(true)
    const view = render(<RolloutControls onCommand={onCommand} onVerify={vi.fn()} phase="paused" />)

    expect(screen.queryByRole('button', { name: 'Pause' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Resume' }))
    expect(onCommand).toHaveBeenCalledWith('resume')
    expect(screen.getByRole('button', { name: 'Resume' })).toHaveProperty('disabled', true)

    view.rerender(<RolloutControls onCommand={onCommand} onVerify={vi.fn()} phase="running" />)
    expect(screen.queryByRole('button', { name: 'Resume' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Pause' })).toBeTruthy()
  })

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

    expect(globalThis.document.activeElement).toBe(pause)
    expect(screen.getByRole('button', { name: 'Pausing…' })).toHaveProperty('disabled', true)
    expect(onCommand).toHaveBeenCalledWith('pause')
    expect(onSubmit).not.toHaveBeenCalled()
  })
})
