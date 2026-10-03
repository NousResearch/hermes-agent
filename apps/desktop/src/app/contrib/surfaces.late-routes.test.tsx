/**
 * A plugin page route registered AFTER the workspace surface mounts must
 * become navigable as a route tile. Regression for late-loaded desktop plugins (disk plugins
 * load async): the route table was compiled into a memo slot keyed on
 * unrelated props, so a late `routes`-area registration never entered the
 * table — the sidebar row rendered but navigating to the path fell through
 * to the `:sessionId` chat route. Uses the REAL useContributions + registry
 * (unlike surfaces.test.tsx) because the reactive flow is the subject.
 */
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { ChatRoutesSurface } from './surfaces'
import type { WiringActions } from './types'

const { openRouteTile } = vi.hoisted(() => ({ openRouteTile: vi.fn() }))

vi.mock('@/store/connections', () => ({ $activeConnectionId: atom('local') }))
vi.mock('@/store/gateway', () => ({ $gateway: atom<unknown>(null) }))
vi.mock('@/store/profile', () => ({ $activeGatewayProfile: atom('default') }))
vi.mock('@/store/route-tiles', () => ({ openRouteTile }))
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
vi.mock('./latest-actions', () => ({ latestChatActions: () => ({}), latestSidebarActions: () => ({}) }))
vi.mock('./panes', () => ({ setStatusbarItemGroup: vi.fn(), useStatusbarContributions: () => [] }))
vi.mock('../shell/model-menu-panel', () => ({ ModelMenuPanel: () => null }))
vi.mock('../shell/reasoning-menu-panel', () => ({ ReasoningMenuPanel: () => null }))

afterEach(() => {
  cleanup()
  openRouteTile.mockClear()
})

describe('ChatRoutesSurface late-registered plugin routes', () => {
  it('opens a route tile for a page whose route registers after mount', async () => {
    const actions = {} as unknown as WiringActions

    render(
      <MemoryRouter initialEntries={['/late-plugin']}>
        <ChatRoutesSurface actions={actions} />
      </MemoryRouter>
    )

    // Before registration the path falls through to the chat catch-all.
    expect(screen.queryByTestId('late-page')).toBeNull()
    expect(screen.getByTestId('chat-view')).toBeTruthy()

    let dispose = () => {}
    act(() => {
      dispose = registry.register({
        area: 'routes',
        id: 'late-plugin:page',
        data: { path: '/late-plugin' },
        render: () => <div data-testid="late-page" />
      })
    })

    // The late registration must reach the route table and win over the
    // `:sessionId` dynamic route, without ever mounting the plugin page as the
    // workspace's full-screen route content.
    await waitFor(() => expect(openRouteTile).toHaveBeenCalledWith('/late-plugin', 'center'))
    expect(screen.queryByTestId('late-page')).toBeNull()
    expect(screen.getByTestId('chat-view')).toBeTruthy()

    act(() => dispose())
  })
})
