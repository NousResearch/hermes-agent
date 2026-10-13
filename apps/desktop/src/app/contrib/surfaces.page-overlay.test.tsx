/**
 * A full page (Capabilities / Messaging / Artifacts / a contributed route)
 * renders as an OVERLAY above the session view, never in its place. The chat
 * stays MOUNTED under the page, and any navigation back to a session route
 * drops the page layer and shows that session — so a page can never outlive its
 * route (the reported bug: the page stuck after opening another session).
 */
import { act, cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { useEffect } from 'react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ChatRoutesSurface } from './surfaces'
import type { WiringActions } from './types'

// A mount counter proves "not unmounted" — a DOM-identity check alone would pass
// even if React tore the chat down and rebuilt it in the same tick.
const chat = vi.hoisted(() => ({ mounts: 0 }))

vi.mock('@/contrib/react/use-contributions', () => ({ useContributions: vi.fn() }))
vi.mock('@/store/connections', () => ({ $activeConnectionId: atom('local') }))
vi.mock('@/store/gateway', () => ({ $gateway: atom<unknown>(null) }))
vi.mock('@/store/profile', () => ({ $activeGatewayProfile: atom('default') }))
vi.mock('@/store/session', () => ({
  $freshDraftReady: atom(false),
  $gatewayState: atom('open')
}))
vi.mock('../chat', () => ({
  ChatView: () => {
    useEffect(() => {
      chat.mounts += 1
    }, [])

    return <div data-testid="chat-view" />
  }
}))
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

let navigateTo: (to: string) => void = () => {}

function NavigateProbe() {
  navigateTo = useNavigate()

  return null
}

function renderAt(pathname: string) {
  const actions = {} as unknown as WiringActions

  return render(
    <MemoryRouter initialEntries={[pathname]}>
      <NavigateProbe />
      <ChatRoutesSurface actions={actions} />
    </MemoryRouter>
  )
}

afterEach(() => {
  cleanup()
  chat.mounts = 0
})

describe('a full page is an overlay above the session view', () => {
  it('keeps the session view mounted when a page route opens', async () => {
    renderAt('/session-a')

    expect(screen.getByTestId('chat-view')).toBeTruthy()
    expect(chat.mounts).toBe(1)

    act(() => navigateTo('/capabilities'))

    // The page is up, and the chat is still there — never unmounted, never
    // remounted.
    expect(await screen.findByTestId('page-view')).toBeTruthy()
    expect(screen.getByTestId('chat-view')).toBeTruthy()
    expect(chat.mounts).toBe(1)
  })

  it('renders the page in a layer above the chat, covering the same area', async () => {
    renderAt('/capabilities')

    const page = await screen.findByTestId('page-view')
    const overlay = page.closest('.absolute')

    expect(overlay).not.toBeNull()
    expect(overlay!.className).toContain('inset-0')

    // The overlay follows the chat in document order, so with both painted in
    // the pane it sits on top.
    const chatView = screen.getByTestId('chat-view')
    expect(chatView.compareDocumentPosition(overlay!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('drops the page layer on any session-route navigation and never brings it back', async () => {
    renderAt('/capabilities')

    expect(await screen.findByTestId('page-view')).toBeTruthy()
    expect(chat.mounts).toBe(1)

    // Open another session: the page goes, that session shows.
    act(() => navigateTo('/session-b'))
    expect(screen.queryByTestId('page-view')).toBeNull()
    expect(screen.getByTestId('chat-view')).toBeTruthy()

    // Return to the original session: the SESSION shows, not the page.
    act(() => navigateTo('/session-a'))
    expect(screen.queryByTestId('page-view')).toBeNull()
    expect(screen.getByTestId('chat-view')).toBeTruthy()
    expect(chat.mounts).toBe(1)
  })

  it('drops the page layer on the new-chat route', async () => {
    renderAt('/artifacts')

    expect(await screen.findByTestId('page-view')).toBeTruthy()

    act(() => navigateTo('/'))
    expect(screen.queryByTestId('page-view')).toBeNull()
    expect(screen.getByTestId('chat-view')).toBeTruthy()
  })
})
