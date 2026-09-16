import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { MemoryRouter, useLocation, useNavigate } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { $gateway } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'

import { ChatRoutesSurface } from './surfaces'
import type { WiringActions } from './types'

const { contributed, openRouteTile } = vi.hoisted(() => ({
  contributed: [] as Array<{ key: string; path: string; render: () => null }>,
  openRouteTile: vi.fn()
}))

vi.mock('@/contrib/react/use-contributions', () => ({ useContributions: vi.fn() }))
vi.mock('@/store/route-tiles', () => ({ openRouteTile }))
vi.mock('@/store/connections', () => ({ $activeConnectionId: atom('local') }))
vi.mock('@/store/gateway', () => ({ $gateway: atom<unknown>(null) }))
vi.mock('@/store/profile', () => ({ $activeGatewayProfile: atom('default') }))
vi.mock('@/store/session', () => ({
  $freshDraftReady: atom(false),
  $gatewayState: atom('open')
}))
vi.mock('../chat', () => ({
  ChatView: ({ gateway }: { gateway: { id?: string } | null }) => <div data-testid="gateway">{gateway?.id}</div>
}))
vi.mock('../chat/sidebar', () => ({ ChatSidebar: () => null }))
vi.mock('../right-sidebar/terminal/chrome', () => ({ TerminalPaneChrome: () => null }))
vi.mock('../shell/hooks/use-status-snapshot', () => ({ useStatusSnapshot: () => ({}) }))
vi.mock('../shell/hooks/use-statusbar-items', () => ({
  useStatusbarItems: () => ({ leftStatusbarItems: [], statusbarItems: [] })
}))
vi.mock('../shell/statusbar-controls', () => ({ StatusbarControls: () => null }))
vi.mock('../routes', () => ({
  contributedRoutes: () => contributed,
  NEW_CHAT_ROUTE: '/new',
  ROUTES_AREA: 'routes',
  sessionRoute: (id: string) => `/${id}`
}))
vi.mock('./latest-actions', () => ({ latestChatActions: () => ({}), latestSidebarActions: () => ({}) }))
vi.mock('./panes', () => ({ setStatusbarItemGroup: vi.fn(), useStatusbarContributions: () => [] }))
vi.mock('../shell/model-menu-panel', () => ({ ModelMenuPanel: () => null }))
vi.mock('../shell/reasoning-menu-panel', () => ({ ReasoningMenuPanel: () => null }))

afterEach(() => {
  cleanup()
  $gateway.set(null)
  $activeGatewayProfile.set('default')
  contributed.length = 0
  openRouteTile.mockClear()
})

/** Exposes the router to the test: current pathname + a navigate handle. */
let probeNavigate: ReturnType<typeof useNavigate> | null = null

function RouterProbe() {
  probeNavigate = useNavigate()

  return <div data-testid="pathname">{useLocation().pathname}</div>
}

function renderWithRoutes(initialEntries: string[]) {
  const actions = { getGateway: () => $gateway.get() } as unknown as WiringActions

  contributed.push({ key: 'plugin:kanban:page', path: '/kanban', render: () => null })

  return render(
    <MemoryRouter initialEntries={initialEntries}>
      <RouterProbe />
      <ChatRoutesSurface actions={actions} />
    </MemoryRouter>
  )
}

describe('ChatRoutesSurface — contributed routes land as route tiles', () => {
  it('opens the tile as a workspace tab and steps back to where the user was on a pushed navigation', async () => {
    renderWithRoutes(['/session-a'])

    expect(screen.getByTestId('pathname').textContent).toBe('/session-a')

    // What `host.navigate('/kanban')` (a titlebar widget, a palette row) does.
    act(() => {
      probeNavigate!('/kanban')
    })

    await waitFor(() => expect(screen.getByTestId('pathname').textContent).toBe('/session-a'))
    expect(openRouteTile).toHaveBeenCalledTimes(1)
    expect(openRouteTile).toHaveBeenCalledWith('/kanban', 'center')
  })

  it('lands on the new chat when there is nothing to step back to (cold start / replaced entry)', async () => {
    renderWithRoutes(['/kanban'])

    await waitFor(() => expect(screen.getByTestId('pathname').textContent).toBe('/new'))
    expect(openRouteTile).toHaveBeenCalledTimes(1)
    expect(openRouteTile).toHaveBeenCalledWith('/kanban', 'center')
  })
})

describe('ChatRoutesSurface', () => {
  it('passes the live gateway after an open-to-open profile switch', () => {
    const gatewayA = { id: 'a' } as unknown as HermesGateway
    const gatewayB = { id: 'b' } as unknown as HermesGateway

    $gateway.set(gatewayA)
    const actions = { getGateway: () => $gateway.get() } as unknown as WiringActions

    render(
      <MemoryRouter>
        <ChatRoutesSurface actions={actions} />
      </MemoryRouter>
    )

    expect(screen.getByTestId('gateway').textContent).toBe('a')

    act(() => {
      $gateway.set(gatewayB)
      $activeGatewayProfile.set('other')
    })

    expect(screen.getByTestId('gateway').textContent).toBe('b')
  })
})
