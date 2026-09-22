import { renderHook } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection, $currentCwd, $selectedStoredSessionId, $sessions } from '@/store/session'
import { $focusedTreePaneId as $focusedTreePaneIdMock } from '@/store/session-focus'
import { $sessionTiles } from '@/store/session-states'

import { useStatusbarItems } from './use-statusbar-items'

// The mock above replaces the computed store with a writable atom.
const $focusedTreePaneId = $focusedTreePaneIdMock as unknown as WritableAtom<null | string>

// The focused pane is derived from the layout tree; a settable atom stands in
// so a test can focus a tile without building a pane tree.
vi.mock('@/store/session-focus', async () => {
  const { atom } = await import('nanostores')

  return { $focusedTreePaneId: atom<null | string>(null) }
})

const wrapper = ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter>

function workspaceMenuIds(): string[] {
  const { result } = renderHook(
    () =>
      useStatusbarItems({
        agentsOpen: false,
        chatOpen: true,
        commandCenterOpen: false,
        extraLeftItems: [],
        extraRightItems: [],
        freshDraftReady: false,
        gatewayState: 'ready',
        inferenceStatus: null,
        openAgents: () => {},
        openCommandCenterSection: () => {},
        requestGateway: async () => undefined as never,
        statusSnapshot: null,
        toggleCommandCenter: () => {}
      }),
    { wrapper }
  )

  const workspace = result.current.leftStatusbarItems.find(item => item.id === 'workspace-cwd')

  return (workspace?.menuItems ?? []).map(item => item.id)
}

afterEach(() => {
  $connection.set(null)
  $currentCwd.set('')
  $sessionTiles.set([])
  $focusedTreePaneId.set(null)
  $selectedStoredSessionId.set(null)
  $sessions.set([])
})

describe('statusbar workspace menu — "Open containing folder"', () => {
  it('offers reveal for a local workspace and hides it when the connection is remote', () => {
    $currentCwd.set('/home/me/project')
    $connection.set({ mode: 'local' } as never)
    expect(workspaceMenuIds()).toContain('reveal-workspace-finder')

    $connection.set({ mode: 'remote' } as never)
    expect(workspaceMenuIds()).not.toContain('reveal-workspace-finder')
  })

  // The reporter's topology (#115167): the window's primary is local but the
  // focused tile is a Connections-tagged session on a remote gateway. The gate
  // follows the FOCUSED session's owner, not the window's primary.
  it('hides reveal for a focused remote tile inside a local-primary window', () => {
    $connection.set({ mode: 'local' } as never)
    $selectedStoredSessionId.set('primary-local')
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'conn-remote', mode: 'remote', profile: 'default' },
        storedSessionId: 'tile-remote'
      }
    ])
    $focusedTreePaneId.set('session-tile:tile-remote')
    // A focused tile never inherits the primary's cwd; its stored row carries it.
    $sessions.set([{ cwd: '/srv/bot/workspace', id: 'tile-remote' }] as never)

    expect(workspaceMenuIds()).toContain('copy-workspace-path')
    expect(workspaceMenuIds()).not.toContain('reveal-workspace-finder')
  })
})
