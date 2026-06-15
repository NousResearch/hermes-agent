import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'

import { StatusbarControls } from './statusbar-controls'

describe('StatusbarControls', () => {
  afterEach(() => {
    cleanup()
  })

  it('applies title and aria-label to action buttons', () => {
    render(
      <MemoryRouter>
        <StatusbarControls
          items={[
            {
              id: 'approval-bypass',
              label: 'Approvals',
              detail: 'Manual',
              onSelect: () => {},
              title: 'YOLO off — click to auto-approve dangerous commands.',
              variant: 'action'
            }
          ]}
        />
      </MemoryRouter>
    )

    const control = screen.getByRole('button', {
      name: 'YOLO off — click to auto-approve dangerous commands.'
    })

    expect(control.getAttribute('title')).toBe('YOLO off — click to auto-approve dangerous commands.')
    expect(screen.getByText('Approvals')).toBeTruthy()
    expect(screen.getByText('Manual')).toBeTruthy()
  })
})
