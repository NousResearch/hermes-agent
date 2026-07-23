import { cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { I18nProvider } from '@/i18n'
import { $routeTiles, closeRouteTile } from '@/store/route-tiles'
import { $selectedStoredSessionId } from '@/store/session'

import { NEW_CHAT_ROUTE, ROUTES_AREA, sessionRoute } from '../routes'

import { TitlebarControls } from './titlebar-controls'

function LocationProbe() {
  const location = useLocation()

  return <div data-testid="location">{location.pathname}</div>
}

describe('TitlebarControls route pane affordance', () => {
  afterEach(() => {
    cleanup()
    closeRouteTile('/kanban')
    $selectedStoredSessionId.set(null)
  })

  it('opens the current contributed route as a route tile and returns to the focused stored session route', () => {
    const dispose = registry.register({
      area: ROUTES_AREA,
      data: { path: '/kanban' },
      id: 'route',
      render: () => null,
      title: 'Kanban'
    })

    try {
      const focusedSessionId = 'focused-session'
      $selectedStoredSessionId.set(focusedSessionId)

      const result = render(
        <I18nProvider configClient={null} initialLocale="en">
          <MemoryRouter initialEntries={['/kanban']}>
            <TitlebarControls onOpenSettings={() => undefined} />
            <LocationProbe />
          </MemoryRouter>
        </I18nProvider>
      )

      fireEvent.click(result.getByRole('button', { name: 'Open Kanban in pane' }))

      expect($routeTiles.get()).toContainEqual({ dir: 'right', path: '/kanban' })
      expect(result.getByTestId('location').textContent).toBe(sessionRoute(focusedSessionId))
    } finally {
      dispose()
    }
  })

  it('opens the current contributed route as a route tile and returns to the default chat route when no focused session exists', () => {
    const dispose = registry.register({
      area: ROUTES_AREA,
      data: { path: '/kanban' },
      id: 'route',
      render: () => null,
      title: 'Kanban'
    })

    try {
      const result = render(
        <I18nProvider configClient={null} initialLocale="en">
          <MemoryRouter initialEntries={['/kanban']}>
            <TitlebarControls onOpenSettings={() => undefined} />
            <LocationProbe />
          </MemoryRouter>
        </I18nProvider>
      )

      fireEvent.click(result.getByRole('button', { name: 'Open Kanban in pane' }))

      expect($routeTiles.get()).toContainEqual({ dir: 'right', path: '/kanban' })
      expect(result.getByTestId('location').textContent).toBe(NEW_CHAT_ROUTE)
    } finally {
      dispose()
    }
  })
})
