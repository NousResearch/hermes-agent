import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { APP_ROUTES, isOverlayView } from '../routes'

import { ChatRoutesSurface } from './surfaces'
import type { WiringActions } from './types'

vi.mock('@/contrib/react/use-contributions', () => ({ useContributions: vi.fn() }))
vi.mock('@/store/connections', () => ({ $activeConnectionId: atom('local') }))
vi.mock('@/store/gateway', () => ({ $gateway: atom<unknown>(null) }))
vi.mock('@/store/profile', () => ({ $activeGatewayProfile: atom('default') }))
vi.mock('@/store/session', () => ({
  $freshDraftReady: atom(false),
  $gatewayState: atom('open')
}))
vi.mock('../chat', () => ({ ChatView: () => <div data-testid="chat-view" /> }))
vi.mock('../capabilities', () => ({ CapabilitiesView: () => null }))
vi.mock('../messaging', () => ({ MessagingView: () => null }))
vi.mock('../artifacts', () => ({ ArtifactsView: () => null }))
vi.mock('../chat/sidebar', () => ({ ChatSidebar: () => null }))
vi.mock('../right-sidebar/terminal/chrome', () => ({ TerminalPaneChrome: () => null }))
vi.mock('../shell/hooks/use-status-snapshot', () => ({ useStatusSnapshot: () => ({}) }))
vi.mock('../shell/hooks/use-statusbar-items', () => ({
  useStatusbarItems: () => ({ leftStatusbarItems: [], statusbarItems: [] })
}))
vi.mock('../shell/statusbar-controls', () => ({ StatusbarControls: () => null }))
vi.mock('./latest-actions', () => ({ latestChatActions: () => ({}), latestSidebarActions: () => ({}) }))
vi.mock('./panes', () => ({ setStatusbarItemGroup: vi.fn(), useStatusbarContributions: () => [] }))
vi.mock('../shell/model-menu-panel', () => ({ ModelMenuPanel: () => null }))
vi.mock('../shell/reasoning-menu-panel', () => ({ ReasoningMenuPanel: () => null }))

afterEach(cleanup)

// The route table (`APP_ROUTES`) and the `<Route>` list are kept by hand in two
// files. A destination missing from the `<Route>` list matches `:sessionId`, so
// the app opens a chat for a session named after the page. Every non-chat
// destination is now an overlay card — no page replaces the workspace pane.
describe('ChatRoutesSurface and APP_ROUTES', () => {
  const destinations = APP_ROUTES.filter(route => route.view !== 'chat')
  const surfaceDestinations = destinations.filter(route => route.id !== 'session-import')

  it.each(destinations.map(route => [route.path, route.view]))('%s is classified as an overlay', (_path, view) => {
    expect(isOverlayView(view)).toBe(true)
  })

  // Session import is mounted by the shell's overlay host, not by this
  // workspace surface. The remaining destinations must still match an
  // explicit route here rather than falling through to `:sessionId`.
  it.each(surfaceDestinations.map(route => [route.path]))('%s never opens as a session', path => {
    const actions = { getGateway: () => null } as unknown as WiringActions

    render(
      <MemoryRouter initialEntries={[path]}>
        <ChatRoutesSurface actions={actions} />
      </MemoryRouter>
    )

    expect(screen.queryByTestId('chat-view')).toBeNull()
  })
})
