import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $sessions, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

vi.mock('./sidebar/session-actions-menu', async () => {
  const React = await import('react')

  return {
    SessionContextMenu: ({ children, connectionId, profile }: { children: React.ReactNode; connectionId?: string; profile?: string }) =>
      React.createElement('div', { 'data-testid': 'tab-menu', 'data-connection': connectionId, 'data-profile': profile }, children)
  }
})

const { SessionTabMenu } = await import('./session-tile')

function tab() {
  render(<SessionTabMenu storedSessionId="s1" tabPaneId="session-tile:s1"><button type="button">Tab</button></SessionTabMenu>)

  return screen.getByTestId('tab-menu')
}

afterEach(() => {
  cleanup()
  $sessions.set([])
  $sessionTiles.set([])
  _resetSessionOwnerHintsForTests()
})

describe('SessionTabMenu Slack consent owner', () => {
  it('uses the explicit tab route instead of the sole same-ID local row', () => {
    $sessions.set([{ id: 's1', profile: 'default', connection_id: 'local', slack_sync_available: true } as never])
    $sessionTiles.set([{ storedSessionId: 's1', ownerRoute: { connectionId: 'remote-a', profile: 'work' } }])
    const menu = tab()
    expect(menu.getAttribute('data-connection')).toBe('remote-a')
    expect(menu.getAttribute('data-profile')).toBe('work')
  })

  it('hides consent when the tab has no proven owner, even with a sole matching row', () => {
    $sessions.set([{ id: 's1', profile: 'default', connection_id: 'local', slack_sync_available: true } as never])
    $sessionTiles.set([{ storedSessionId: 's1' }])
    const menu = tab()
    expect(menu.getAttribute('data-connection')).toBeNull()
    expect(menu.getAttribute('data-profile')).toBe('default')
  })

  it('passes the exact owner of a legitimate tab', () => {
    $sessions.set([{ id: 's1', profile: 'work', connection_id: 'remote-a', slack_sync_available: true } as never])
    setSessionOwnerHint('s1', { connectionId: 'remote-a', profile: 'work' })
    $sessionTiles.set([{ storedSessionId: 's1', ownerRoute: { connectionId: 'remote-a', profile: 'work' } }])
    const menu = tab()
    expect(menu.getAttribute('data-connection')).toBe('remote-a')
    expect(menu.getAttribute('data-profile')).toBe('work')
  })
})
