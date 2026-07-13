import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { StatusbarControls } from './statusbar-controls'

describe('StatusbarControls', () => {
  it('gives icon-only actions and menus accessible names from their titles', () => {
    render(
      <MemoryRouter>
        <StatusbarControls
          items={[
            {
              icon: <span aria-hidden="true">A</span>,
              id: 'action',
              title: 'Open command center',
              variant: 'action'
            },
            {
              icon: <span aria-hidden="true">M</span>,
              id: 'menu',
              menuItems: [{ id: 'item', label: 'Menu item' }],
              title: 'Open gateway menu',
              variant: 'menu'
            }
          ]}
        />
      </MemoryRouter>
    )

    expect(screen.getByRole('button', { name: 'Open command center' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open gateway menu' })).toBeTruthy()
  })

  it('uses a visible string label as the accessible name', () => {
    render(
      <MemoryRouter>
        <StatusbarControls
          items={[
            {
              id: 'cron',
              label: 'Scheduled tasks',
              title: 'Open scheduled tasks',
              variant: 'action'
            }
          ]}
        />
      </MemoryRouter>
    )

    expect(screen.getByRole('button', { name: 'Scheduled tasks' })).toBeTruthy()
  })
})
