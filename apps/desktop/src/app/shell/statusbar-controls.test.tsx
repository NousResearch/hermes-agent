import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { StatusbarControls } from './statusbar-controls'

afterEach(() => {
  cleanup()
})

describe('StatusbarControls', () => {
  it('applies status item titles as native tooltips', () => {
    render(
      <MemoryRouter>
        <StatusbarControls items={[{ id: 'context', label: 'Context', title: 'Context usage', variant: 'text' }]} />
      </MemoryRouter>
    )

    expect(screen.getByText('Context').closest('div')?.getAttribute('title')).toBe('Context usage')
  })

  it('uses the title as the accessible name for icon-only actions', () => {
    const onSelect = vi.fn()

    render(
      <MemoryRouter>
        <StatusbarControls
          items={[
            {
              icon: <span aria-hidden="true">bolt</span>,
              id: 'yolo',
              onSelect,
              title: 'YOLO off - click to auto-approve dangerous commands.',
              variant: 'action'
            }
          ]}
        />
      </MemoryRouter>
    )

    const button = screen.getByRole('button', { name: 'YOLO off - click to auto-approve dangerous commands.' })

    expect(button.getAttribute('title')).toBe('YOLO off - click to auto-approve dangerous commands.')
  })
})
