import { act, cleanup, render as renderUi, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it } from 'vitest'

import { $interfaceMode } from '@/store/interface-mode'
import { $activeGatewayProfile, $newChatProfile, $newChatRoute, $profiles } from '@/store/profile'
import { $projectTree } from '@/store/projects'
import { $connection, $sessions } from '@/store/session'
import { $botChatScopes } from '@/store/session-states'
import { makeSessionInfo } from '@/test/session-info'

import { SimpleChatContext as ChatContext, ChatContext as ComposerChatContext } from './chat-context'
import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from './composer/scope'

const render = (ui: Parameters<typeof renderUi>[0]) => renderUi(ui, { wrapper: MemoryRouter })

afterEach(() => {
  cleanup()
  $interfaceMode.set('advanced')
  $activeGatewayProfile.set('default')
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  $profiles.set([])
  $projectTree.set([])
  $connection.set(null)
  $botChatScopes.set({})
  $sessions.set([])
})

it('restores a retained chat’s folder from its exact stored owner until runtime context arrives', () => {
  $interfaceMode.set('simple')
  $sessions.set([
    makeSessionInfo({ id: 'retained', profile: 'research', connection_id: 'local', cwd: '/wrong/local-folder' }),
    makeSessionInfo({ id: 'retained', profile: 'research', connection_id: 'remote', cwd: '/work/release' })
  ])

  const view = render(
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, connectionId: 'remote', profile: 'research' }}>
      <ChatContext cwd="" profile="default" storedSessionId="retained" />
    </ComposerScopeProvider>
  )

  expect(screen.getByText('Folder · release').closest('dd')?.getAttribute('title')).toBe('/work/release')
  expect(screen.queryByText('local-folder')).toBeNull()
  view.rerender(
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, connectionId: 'remote', profile: 'research' }}>
      <ChatContext cwd="/runtime/current" profile="default" storedSessionId="retained" />
    </ComposerScopeProvider>
  )
  expect(screen.getByText('Folder · current').closest('dd')?.getAttribute('title')).toBe('/runtime/current')
})

it('shows the Bot main conversation owner even when its composer keeps the ambient profile', () => {
  $interfaceMode.set('simple')
  $botChatScopes.set({
    bot: {
      workspaceMode: 'bots',
      ownerProfile: 'research',
      ownerRoute: { connectionId: 'remote', profile: 'research' }
    }
  })
  render(<ChatContext cwd="" profile="default" storedSessionId="bot" />)
  expect(screen.getByText('research')).toBeTruthy()
  expect(screen.queryByText('No project')).toBeNull()
  expect(screen.queryByText('default')).toBeNull()
})

it('keeps a retained chat’s profile and folder independent of the sidebar’s active owner', () => {
  $interfaceMode.set('simple')
  $activeGatewayProfile.set('default')
  $projectTree.set([{ id: 'p_local', label: 'Local project', path: '/work/repo', repos: [], sessionCount: 0 }])
  render(
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, connectionId: 'remote', profile: 'research' }}>
      <ChatContext cwd="/work/repo" profile="default" storedSessionId="retained" />
    </ComposerScopeProvider>
  )
  expect(screen.getByText('research')).toBeTruthy()
  expect(screen.getByText('Folder · repo')).toBeTruthy()
  expect(screen.queryByText('Local project')).toBeNull()
  act(() => $activeGatewayProfile.set('work'))
  expect(screen.getByText('research')).toBeTruthy()
})

it('updates the current project name, displays an explicit detached state, and stays out of other layouts', () => {
  $interfaceMode.set('simple')
  $projectTree.set([{ id: '/work/repo', label: 'Release work', path: '/work/repo', repos: [], sessionCount: 0 }])
  $sessions.set([
    makeSessionInfo({ id: 'main', profile: 'default', cwd: '/work/repo/subfolder', git_repo_root: '/work/repo' })
  ])
  const view = render(<ChatContext cwd="/work/repo/subfolder" profile="default" storedSessionId="main" />)
  expect(screen.getByText('Release work').closest('dd')?.getAttribute('title')).toBe('/work/repo/subfolder')
  act(() => $projectTree.set([{ id: '/work/repo', label: 'Desktop', path: '/work/repo', repos: [], sessionCount: 0 }]))
  expect(screen.getByText('Desktop')).toBeTruthy()
  view.rerender(<ChatContext cwd="" profile="default" storedSessionId={null} />)
  expect(screen.getByText('No project')).toBeTruthy()
  act(() => $newChatRoute.set({ connectionId: 'remote', profile: 'research' }))
  expect(screen.getByText('research')).toBeTruthy()
  view.rerender(<ComposerChatContext cwd="" profile="default" shown={false} />)
  expect(screen.queryByText('research')).toBeNull()
})

it('does not turn the current working folder into project membership for an unassigned stored chat', () => {
  $interfaceMode.set('simple')
  $sessions.set([makeSessionInfo({ id: 'meeting', profile: 'default', cwd: null })])
  $projectTree.set([{ id: '/work/repo', label: 'Release work', path: '/work/repo', repos: [], sessionCount: 0 }])
  render(<ChatContext cwd="/work/repo" profile="default" storedSessionId="meeting" />)
  expect(screen.queryByText('No project')).toBeNull()
  expect(screen.queryByText('Release work')).toBeNull()
})

it('honors an authoritative unassigned bucket over a matching stored folder', () => {
  $interfaceMode.set('simple')
  $sessions.set([makeSessionInfo({ id: 'meeting', profile: 'default', cwd: '/work/repo' })])
  $projectTree.set([
    { id: '/work/repo', label: 'Release work', path: '/work/repo', repos: [], sessionCount: 0 },
    {
      id: '__no_project__',
      label: 'Home',
      isNoProject: true,
      path: null,
      repos: [],
      sessionCount: 1,
      sessionIds: ['meeting']
    }
  ])
  render(<ChatContext cwd="/work/repo" profile="default" storedSessionId="meeting" />)
  expect(screen.queryByText('No project')).toBeNull()
  expect(screen.queryByText('Release work')).toBeNull()
  expect(screen.queryByText('Home')).toBeNull()
})
