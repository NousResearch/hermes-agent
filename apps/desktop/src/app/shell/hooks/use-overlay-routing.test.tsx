import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useLocation, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { ROUTES_AREA } from '@/app/routes'
import { registry } from '@/contrib/registry'
import { $activeGatewayProfile } from '@/store/profile'
import { setConnection } from '@/store/session'

import { useOverlayRouting } from './use-overlay-routing'

let dispose: () => void

function Harness() {
  const { closeContributedRoute, closeOverlayToPreviousRoute } = useOverlayRouting()
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <>
      <output>
        {location.pathname}
        {location.search}
        {location.hash}
      </output>
      <button onClick={() => void navigate('/kanban')}>Board</button>
      <button onClick={() => void navigate('/settings')}>Settings</button>
      <button onClick={closeOverlayToPreviousRoute}>Close settings</button>
      <button onClick={closeContributedRoute}>Close board</button>
    </>
  )
}

beforeEach(() => {
  setConnection(null)
  $activeGatewayProfile.set('default')
  dispose = registry.register({
    area: ROUTES_AREA,
    id: 'kanban-close-test',
    data: { path: '/kanban' },
    render: () => null
  })
})

afterEach(() => {
  cleanup()
  dispose()
  setConnection(null)
  $activeGatewayProfile.set('default')
})

const click = (name: string) => fireEvent.click(screen.getByRole('button', { name }))
const route = () => screen.getByRole('status').textContent

describe('closing contributed routes', () => {
  it('returns to the full original route after a settings round trip, without a return loop', () => {
    render(
      <MemoryRouter initialEntries={['/conversation?panel=tools#item']}>
        <Harness />
      </MemoryRouter>
    )
    click('Board')
    click('Settings')
    click('Close settings')
    expect(route()).toBe('/kanban')
    click('Close board')
    expect(route()).toBe('/conversation?panel=tools#item')
    click('Board')
    click('Close board')
    expect(route()).toBe('/conversation?panel=tools#item')
  })

  it('returns home when launched directly, including when its previous plugin unloads', () => {
    const removePreviousPage = registry.register({
      area: ROUTES_AREA,
      id: 'previous-page',
      data: { path: '/previous-plugin' },
      render: () => null
    })

    render(
      <MemoryRouter initialEntries={['/previous-plugin']}>
        <Harness />
      </MemoryRouter>
    )
    click('Board')
    removePreviousPage()
    click('Close board')
    expect(route()).toBe('/')
    cleanup()
    render(
      <MemoryRouter initialEntries={['/kanban']}>
        <Harness />
      </MemoryRouter>
    )
    click('Close board')
    expect(route()).toBe('/')
  })

  it.each(['profile', 'connection'] as const)('discards the return route after a %s switch', scope => {
    render(
      <MemoryRouter initialEntries={['/conversation']}>
        <Harness />
      </MemoryRouter>
    )
    click('Board')
    act(() => {
      if (scope === 'profile') {
        $activeGatewayProfile.set('other')
      } else {
        setConnection({
          connectionId: 'other',
          baseUrl: 'https://other.example',
          wsUrl: 'wss://other.example/ws',
          token: '',
          logs: [],
          isFullscreen: false,
          nativeOverlayWidth: 0,
          windowButtonPosition: null
        })
      }
    })
    click('Close board')
    expect(route()).toBe('/')
  })
})
