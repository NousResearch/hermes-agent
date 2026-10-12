import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $interfaceMode } from '@/store/interface-mode'
import { $selectedStoredSessionId, $sessions } from '@/store/session'
import { $botChatScopes } from '@/store/session-states'
import { $openWorkspaceSessionInNewTab } from '@/store/workspace-tab-action'
import { makeSessionInfo } from '@/test/session-info'

import { ChatHeader } from './chat-header'
import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from './composer/scope'
import { SessionActionsMenu, SessionContextMenu } from './sidebar/session-actions-menu'

const openSession = vi.fn()
vi.mock('@/app/open-session', () => ({ openSession: (...args: unknown[]) => openSession(...args) }))

afterEach(() => {
  vi.clearAllMocks()
  cleanup()
  $interfaceMode.set('advanced')
  $sessions.set([])
  $selectedStoredSessionId.set(null)
  $openWorkspaceSessionInNewTab.set(null)
  $botChatScopes.set({})
})

it('renders no header in Simple, where the tab carries the title and actions', () => {
  $interfaceMode.set('simple')
  $sessions.set([makeSessionInfo({ id: 'chat', title: 'Plan the release' })])
  $selectedStoredSessionId.set('chat')

  render(
    <ChatHeader
      activeSessionId="runtime"
      isRoutedSessionView
      onDeleteSelectedSession={vi.fn()}
      onToggleSelectedPin={vi.fn()}
      selectedSessionId="chat"
    />
  )
  expect(screen.queryByText('Plan the release')).toBeNull()
})

it('keeps Advanced current-chat actions unchanged and hides actions for an empty draft', () => {
  $sessions.set([makeSessionInfo({ id: 'chat', title: 'Plan the release' })])
  $selectedStoredSessionId.set('chat')

  const props = {
    activeSessionId: 'runtime',
    selectedSessionId: 'chat',
    isRoutedSessionView: true,
    onDeleteSelectedSession: vi.fn(),
    onToggleSelectedPin: vi.fn()
  }

  const view = render(<ChatHeader {...props} />)
  fireEvent.pointerDown(screen.getByRole('button', { name: 'Plan the release' }), {
    button: 0,
    ctrlKey: false,
    pointerType: 'mouse'
  })
  expect(screen.queryByRole('menuitem', { name: 'Open in new tab' })).toBeNull()
  act(() => $interfaceMode.set('simple'))
  view.rerender(<ChatHeader {...props} activeSessionId={null} isRoutedSessionView={false} selectedSessionId={null} />)
  expect(screen.queryByRole('button', { name: 'Plan the release' })).toBeNull()
})

it.each(['menu', 'context'] as const)('offers the browsing chat in a new tab through the row %s menu', surface => {
  $interfaceMode.set('simple')
  $selectedStoredSessionId.set('chat')
  $openWorkspaceSessionInNewTab.set(vi.fn())
  const Menu = surface === 'context' ? SessionContextMenu : SessionActionsMenu
  render(
    <Menu sessionId="chat" title="Release plan">
      <button type="button">Chat row</button>
    </Menu>
  )
  const row = screen.getByRole('button', { name: 'Chat row' })

  if (surface === 'context') {
    fireEvent.contextMenu(row)
  } else {
    fireEvent.pointerDown(row, { button: 0, ctrlKey: false, pointerType: 'mouse' })
  }

  fireEvent.click(screen.getByRole('menuitem', { name: 'Open in new tab' }))
  expect(openSession).toHaveBeenCalledWith('chat', expect.any(Function), 'tab')
})

it('does not borrow a title from another owner with the same stored id', () => {
  $sessions.set([
    makeSessionInfo({ id: 'chat', profile: 'default', connection_id: 'local', title: 'Local conversation' }),
    makeSessionInfo({ id: 'chat', profile: 'research', connection_id: 'remote', title: 'Remote research' })
  ])
  render(
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, connectionId: 'remote', profile: 'research' }}>
      <ChatHeader
        activeSessionId="runtime"
        isRoutedSessionView
        onDeleteSelectedSession={vi.fn()}
        onToggleSelectedPin={vi.fn()}
        selectedSessionId="chat"
      />
    </ComposerScopeProvider>
  )
  expect(screen.getByText('Remote research')).toBeTruthy()
  expect(screen.queryByText('Local conversation')).toBeNull()
})
