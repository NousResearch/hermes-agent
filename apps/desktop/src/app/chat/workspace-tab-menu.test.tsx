import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { $interfaceMode } from '@/store/interface-mode'
import { $selectedStoredSessionId, $sessions } from '@/store/session'
import { $botChatScopes } from '@/store/session-states'
import { makeSessionInfo } from '@/test/session-info'

import { WorkspaceTabMenu } from './session-tile'

afterEach(() => {
  cleanup()
  $interfaceMode.set('advanced')
  $selectedStoredSessionId.set(null)
  $sessions.set([])
  $botChatScopes.set({})
})

it.each(['simple', 'advanced', 'bot'] as const)('offers Close only for the Advanced main-tab behavior: %s', mode => {
  $interfaceMode.set(mode === 'advanced' ? 'advanced' : 'simple')
  $selectedStoredSessionId.set('chat')
  $sessions.set([makeSessionInfo({ id: 'chat', title: 'Release plan' })])

  if (mode === 'bot') {
    $botChatScopes.set({ chat: { workspaceMode: 'bots', workspaceOwnerKey: 'bot:research' } })
  }

  render(
    <WorkspaceTabMenu>
      <button type="button">Workspace tab</button>
    </WorkspaceTabMenu>
  )
  fireEvent.contextMenu(screen.getByRole('button', { name: 'Workspace tab' }))
  expect(Boolean(screen.queryByRole('menuitem', { name: /^Close$/ }))).toBe(mode === 'advanced')
})
