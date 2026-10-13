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
vi.mock('../capabilities', () => ({ CapabilitiesView: () => <div data-testid="page-view" /> }))
vi.mock('../messaging', () => ({ MessagingView: () => <div data-testid="page-view" /> }))
vi.mock('../artifacts', () => ({ ArtifactsView: () => <div data-testid="page-view" /> }))
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
// files. A workspace page whose path is missing from the `<Route>` list matches
// `:sessionId`, so the app opens a chat for a session named after the page.
// Overlays are exempt: they are modal cards over whatever the shell shows.
//
// A page renders as a layer ABOVE the session view now, so "never opens as a
// session" means the page's own route matched — its content renders — while the
// chat stays MOUNTED beneath it, rather than the chat unmounting.
describe('ChatRoutesSurface and APP_ROUTES', () => {
  const pages = APP_ROUTES.filter(route => route.view !== 'chat' && !isOverlayView(route.view))

  it.each(pages.map(route => [route.path]))('%s renders its page above a still-mounted chat', async path => {
    const actions = { getGateway: () => null } as unknown as WiringActions

    render(
      <MemoryRouter initialEntries={[path]}>
        <ChatRoutesSurface actions={actions} />
      </MemoryRouter>
    )

    expect(await screen.findByTestId('page-view')).toBeTruthy()
    expect(screen.getByTestId('chat-view')).toBeTruthy()
  })
})
