import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { StatusbarControls } from './statusbar-controls'

describe('StatusbarControls accessibility', () => {
  it('uses an icon-only action title as its accessible name', () => {
    render(
      <MemoryRouter>
        <StatusbarControls
          items={[
            {
              icon: <span aria-hidden="true">icon</span>,
              id: 'host-vnc',
              onSelect: () => undefined,
              title: 'Show Host VNC',
              variant: 'action'
            }
          ]}
        />
      </MemoryRouter>
    )

    expect(screen.getByRole('button', { name: 'Show Host VNC' })).toBeDefined()
  })
})
