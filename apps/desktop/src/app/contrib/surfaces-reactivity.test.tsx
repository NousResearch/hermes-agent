import { act, cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { ChatRoutesSurface } from './surfaces'
import type { WiringActions } from './types'

// #109063: the workspace surface subscribes to ROUTES_AREA but used to derive
// routes via an independently-called `contributedRoutes()`, which React
// Compiler can keep memoized across the delivered subscription update. The
// registry, `useContributions` and the routes module all stay REAL here — only
// unrelated stores and heavy sibling views are stubbed, because mocking the
// route list is exactly what hid the defect.

vi.mock('@/store/connections', () => ({ $activeConnectionId: atom('local') }))
vi.mock('@/store/gateway', () => ({ $gateway: atom<unknown>(null) }))
vi.mock('@/store/profile', () => ({ $activeGatewayProfile: atom('default') }))
vi.mock('@/store/session', () => ({
  $freshDraftReady: atom(false),
  $gatewayState: atom('open')
}))
vi.mock('../chat', () => ({ ChatView: () => <div data-testid="chat-view" /> }))
vi.mock('../chat/sidebar', () => ({ ChatSidebar: () => null }))
vi.mock('../right-sidebar/terminal/chrome', () => ({ TerminalPaneChrome: () => null }))
vi.mock('../shell/hooks/use-status-snapshot', () => ({ useStatusSnapshot: () => ({}) }))
vi.mock('../shell/hooks/use-statusbar-items', () => ({
  useStatusbarItems: () => ({ leftStatusbarItems: [], statusbarItems: [] })
}))
vi.mock('../shell/statusbar-controls', () => ({ StatusbarControls: () => null }))
vi.mock('../shell/model-menu-panel', () => ({ ModelMenuPanel: () => null }))
vi.mock('./latest-actions', () => ({ latestChatActions: () => ({}), latestSidebarActions: () => ({}) }))
vi.mock('./panes', () => ({ setStatusbarItemGroup: vi.fn(), useStatusbarContributions: () => [] }))

afterEach(cleanup)

/** Register (or hot-replace) the plugin route page mounted at `/resetwatch`. */
function registerResetwatch(label: string) {
  return registry.register({
    id: 'resetwatch',
    area: 'routes',
    title: `Resetwatch ${label}`,
    data: { path: '/resetwatch' },
    render: () => <div>{`PLUGIN-PAGE-${label}`}</div>
  })
}

describe('ChatRoutesSurface route-contribution reactivity (#109063)', () => {
  it('mounts a plugin page registered after the surface, tracks replacement, and unmounts on dispose', () => {
    const actions = { getGateway: () => null } as unknown as WiringActions

    render(
      <MemoryRouter initialEntries={['/resetwatch']}>
        <ChatRoutesSurface actions={actions} />
      </MemoryRouter>
    )
    // Unregistered path falls through the catch-all redirect to the chat view.
    expect(screen.getByTestId('chat-view')).toBeTruthy()

    let dispose = () => {}
    act(() => {
      // Late registration after the surface has mounted and settled.
      dispose = registerResetwatch('A')
    })
    expect(screen.getByText('PLUGIN-PAGE-A')).toBeTruthy()
    expect(screen.queryByTestId('chat-view')).toBeNull()

    act(() => {
      dispose()
      dispose = registerResetwatch('B')
    })
    expect(screen.getByText('PLUGIN-PAGE-B')).toBeTruthy()

    act(() => {
      dispose()
    })
    expect(screen.getByTestId('chat-view')).toBeTruthy()

    act(() => {
      dispose = registerResetwatch('A')
    })
    expect(screen.getByText('PLUGIN-PAGE-A')).toBeTruthy()

    act(() => {
      dispose()
    })
  })
})
