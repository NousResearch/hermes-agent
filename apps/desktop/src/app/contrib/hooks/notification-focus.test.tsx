import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { paneMirror } from '@/app/chat/pane-mirror'
import { sessionRoute, syncWorkspaceRoute } from '@/app/routes'
import { group } from '@/components/pane-shell/tree/model'
import * as tree from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $selectedStoredSessionId } from '@/store/session'
import {
  $sessionTiles,
  closeSessionTile,
  discardSessionTile,
  openSessionTile,
  patchSessionTile
} from '@/store/session-states'

import { useDesktopIntegrations } from './use-desktop-integrations'

const originalBridge = window.hermesDesktop

beforeAll(() => {
  const dispose = registry.register({
    area: 'panes',
    data: { placement: 'main', uncloseable: true },
    id: 'workspace',
    render: () => null,
    title: 'Chat'
  })

  tree.watchContributedPanes()
  paneMirror({
    source: $sessionTiles,
    key: tile => tile.storedSessionId,
    prefix: 'session-tile',
    dir: () => 'center',
    minWidth: '20rem',
    title: id => id,
    render: () => null,
    close: closeSessionTile
  })()

  return dispose
})

beforeEach(() => {
  window.localStorage.clear()
  tree.declareDefaultTree(group(['workspace'], { active: 'workspace', id: 'main' }))
  tree.$layoutTree.set(group(['workspace'], { active: 'workspace', id: 'main' }))
  $selectedStoredSessionId.set('main-chat')
  syncWorkspaceRoute('/main-chat')
})

afterEach(() => {
  for (const tile of $sessionTiles.get()) {
    discardSessionTile(tile.storedSessionId)
  }

  $selectedStoredSessionId.set(null)
  window.hermesDesktop = originalBridge
  syncWorkspaceRoute('/')
})

it('a native click reveals the existing remote Bot tab without changing its owner or duplicating main', () => {
  let fire!: (id: string) => void
  window.hermesDesktop = {
    onFocusSession: callback => {
      fire = callback

      return () => undefined
    }
  } as Window['hermesDesktop']
  const navigate = vi.fn()
  renderHook(() =>
    useDesktopIntegrations({
      activeProfile: 'default',
      chatOpen: false,
      hasPreview: false,
      locationPathname: '/settings',
      navigate,
      profileReady: false,
      refreshSessions: vi.fn(),
      resumeLastSession: false,
      resumeExhaustedSessionId: null,
      routedSessionId: null,
      runtimeIdByStoredSessionId: { current: new Map() },
      sessions: []
    })
  )

  const scope = {
    ownerRoute: { connectionId: 'remote-writer', profile: 'writer', mode: 'remote' as const },
    workspaceMode: 'bots' as const,
    workspaceOwnerKey: 'remote-writer::writer',
    workspaceTabTitle: 'Bot Chat'
  }

  openSessionTile('bot-chat', 'center', 'workspace', undefined, scope)
  patchSessionTile('bot-chat', { runtimeId: 'bot-runtime' })
  tree.revealTreePane('workspace')
  const before = $sessionTiles.get()

  act(() => fire('bot-runtime'))

  expect(tree.isPaneVisible('session-tile:bot-chat')).toBe(true)
  expect($sessionTiles.get()).toEqual(before)
  expect($selectedStoredSessionId.get()).toBe('main-chat')
  // Close the overlay without navigating main onto the tile's conversation.
  expect(navigate).toHaveBeenCalledWith(sessionRoute('main-chat'), { replace: true })
})
