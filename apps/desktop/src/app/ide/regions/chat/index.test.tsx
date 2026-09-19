// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as SessionTileModule from '@/app/chat/session-tile'
import { I18nProvider } from '@/i18n'

const h = vi.hoisted(() => ({
  close: vi.fn(),
  create: vi.fn(),
  list: vi.fn(),
  remember: vi.fn(),
  request: vi.fn()
}))

vi.mock('@/app/chat/session-tile', async importOriginal => {
  const actual = await importOriginal<typeof SessionTileModule>()

  return {
    ...actual,
    requestCloseSessionTile: (...args: unknown[]) => h.close(...args),
    SessionTilePane: ({ storedSessionId }: { storedSessionId: string }) => (
      <div data-testid="tile-pane">{storedSessionId}</div>
    ),
    tileStoredRow: () => undefined
  }
})

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: h.request })
}))

vi.mock('./sessions', () => ({
  createIdeSession: (...args: unknown[]) => h.create(...args),
  listIdeSessions: (...args: unknown[]) => h.list(...args),
  rememberIdeSessionRows: (...args: unknown[]) => h.remember(...args)
}))

import { $gatewayState } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import type { SessionTile } from '@/store/session-states'

import { $ideActiveChat } from './store'

import { ChatRegion } from './index'

const tile = (storedSessionId: string) => ({ storedSessionId }) as SessionTile

function renderRegion() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <ChatRegion />
    </I18nProvider>
  )
}

beforeEach(() => {
  h.close.mockReset()
  h.create.mockReset().mockResolvedValue('s-new')
  h.list.mockReset().mockResolvedValue([])
  h.remember.mockReset()
  h.request.mockReset()
  $sessionTiles.set([])
  $ideActiveChat.set(null)
  $gatewayState.set('open')
})

afterEach(() => {
  cleanup()
})

describe('ChatRegion', () => {
  it('shows the IDE-scoped empty state with a create action', async () => {
    renderRegion()

    expect(screen.getByText('No IDE session yet')).toBeTruthy()
    // The empty state's labeled button (the header's icon-only "+" shares the
    // accessible name but carries no text).
    fireEvent.click(screen.getByText('New IDE session'))

    await waitFor(() => expect(h.create).toHaveBeenCalledTimes(1))
    expect(h.create).toHaveBeenCalledWith(h.request)
  })

  it('renders one tab per open session and mounts the active chat', () => {
    $sessionTiles.set([tile('s1'), tile('s2')])

    renderRegion()

    // No active pointer yet: the first tile wins.
    expect(screen.getByTestId('tile-pane').textContent).toBe('s1')

    fireEvent.click(screen.getAllByText('New session')[1])

    expect(screen.getByTestId('tile-pane').textContent).toBe('s2')
  })

  it('routes tab closes through the guarded tile closer', () => {
    $sessionTiles.set([tile('s1')])

    renderRegion()
    fireEvent.click(screen.getByLabelText('Close New session'))

    expect(h.close).toHaveBeenCalledWith('s1')
  })

  it('lists recent IDE sessions and reopens one as a tab', async () => {
    h.list.mockResolvedValue([{ id: 'r1', title: 'Old chat' }])

    renderRegion()

    const recent = await screen.findByText('Old chat')
    fireEvent.click(recent)

    expect($ideActiveChat.get()).toBe('r1')
    expect($sessionTiles.get().some(entry => entry.storedSessionId === 'r1')).toBe(true)
  })

  it('re-arms the active pointer when its tab disappears', () => {
    $sessionTiles.set([tile('s1'), tile('s2')])
    $ideActiveChat.set('s2')

    const view = renderRegion()

    expect(screen.getByTestId('tile-pane').textContent).toBe('s2')

    // The active tab is closed elsewhere (menu, another surface).
    $sessionTiles.set([tile('s1')])
    view.rerender(
      <I18nProvider configClient={null} initialLocale="en">
        <ChatRegion />
      </I18nProvider>
    )

    expect($ideActiveChat.get()).toBe('s1')
    expect(screen.getByTestId('tile-pane').textContent).toBe('s1')
  })
})
