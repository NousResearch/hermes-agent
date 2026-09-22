// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { EntityKey } from '../model'

import { CameraControls } from './camera-controls'

function key(value: string): EntityKey {
  return value as EntityKey
}

describe('CameraControls', () => {
  it('renders every native camera action and dispatches the shared camera intents', () => {
    const dispatch = vi.fn()
    render(
      <CameraControls
        dispatch={dispatch}
        state={{ focusedEntityKey: key('session:local:worker:1'), following: true }}
      />
    )

    const labels = [
      'Rotate Left',
      'Rotate Right',
      'Tilt Up',
      'Tilt Down',
      'Pan North',
      'Pan South',
      'Pan East',
      'Pan West',
      'Zoom In',
      'Zoom Out',
      'Stop Following',
      'Return to City'
    ]

    for (const label of labels) {
      expect(screen.getByRole('button', { name: label }).tagName).toBe('BUTTON')
    }

    fireEvent.click(screen.getByRole('button', { name: 'Rotate Left' }))
    fireEvent.click(screen.getByRole('button', { name: 'Pan North' }))
    fireEvent.click(screen.getByRole('button', { name: 'Zoom In' }))
    fireEvent.click(screen.getByRole('button', { name: 'Stop Following' }))
    fireEvent.click(screen.getByRole('button', { name: 'Return to City' }))

    expect(dispatch).toHaveBeenCalledWith({ kind: 'orbit', deltaAlpha: -0.22, deltaBeta: 0 })
    expect(dispatch).toHaveBeenCalledWith({ kind: 'pan', deltaX: 0, deltaZ: -3 })
    expect(dispatch).toHaveBeenCalledWith({ kind: 'zoom', delta: -6 })
    expect(dispatch).toHaveBeenCalledWith({ kind: 'focus', entityKey: key('session:local:worker:1'), follow: false })
    expect(dispatch).toHaveBeenCalledWith({ kind: 'return-to-city' })
  })

  it('reports focus and follow changes through one polite live region, not per animation frame', () => {
    const { rerender } = render(
      <CameraControls
        dispatch={vi.fn()}
        focusedEntityLabel="Pip, research worker"
        state={{ focusedEntityKey: key('session:local:worker:1'), following: true }}
      />
    )

    const status = screen.getByRole('status', { name: 'Camera position' })
    expect(status.getAttribute('aria-live')).toBe('polite')
    expect(status.textContent).toBe('Following Pip, research worker')
    expect(status.textContent).not.toContain('session:local:worker:1')

    rerender(
      <CameraControls
        dispatch={vi.fn()}
        focusedEntityLabel="Pip, research worker"
        state={{ focusedEntityKey: key('session:local:worker:1'), following: false }}
      />
    )
    expect(status.textContent).toBe('Focused on Pip, research worker')
  })
})
