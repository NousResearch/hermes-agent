import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useEffect } from 'react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { $workspaceIsPage } from '@/app/routes'
import { StatusbarControls } from '@/app/shell/statusbar-controls'
import { en } from '@/i18n/en'
import { $primaryGatewayRoute } from '@/store/gateway-primary-route'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import {
  $cronSessions,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  $unlistedSessionOwnerRows
} from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import { $shellChoice } from '@/store/shell-variant'
import { $statusbarHiddenIds } from '@/store/statusbar-prefs'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'
import { makeSessionInfo } from '@/test/session-info'

import { PortalConversationHeader, usePortalConversation } from './portal-conversation-header'
import { PortalStatusMenu } from './portal-status-menu'

beforeEach(() => {
  stubResizeObserver()
  stubMenuDomApis()
  $shellChoice.set(null)
  $workspaceIsPage.set(false)
  $apiRequestScope.set({ connectionId: 'local', profile: 'default' })
  $activeGatewayProfile.set('default')
  $profiles.set([{ name: 'default', is_default: true }] as typeof $profiles.value)
})

afterEach(() => {
  cleanup()
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $unlistedSessionOwnerRows.set([])
  $sessionTiles.set([])
  $selectedStoredSessionId.set(null)
  $statusbarHiddenIds.set([])
  $apiRequestScope.set({ connectionId: null, profile: null })
  $primaryGatewayRoute.set({ connectionId: null, mode: null })
})

function Location() {
  return <output>{useLocation().search}</output>
}

it('opens the viewed tile’s agent settings, never the ambient agent', () => {
  $sessions.set([makeSessionInfo({ id: 'remote-chat', connection_id: 'remote', profile: 'research' })])
  render(
    <MemoryRouter>
      <PortalConversationHeader paneId="session-tile:remote-chat" title="Research notes" />
      <Location />
    </MemoryRouter>
  )
  expect(screen.getByRole('heading', { name: 'Research notes' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Agent settings' }))
  expect(screen.getByRole('status').textContent).toBe('?agentProfile=research&connection=remote')
})

it('fails closed for unknown session owners, and resolves cron owners after loading', () => {
  $selectedStoredSessionId.set('scheduled-chat')
  render(
    <MemoryRouter>
      <PortalConversationHeader paneId="workspace" title="Scheduled report" />
      <Location />
    </MemoryRouter>
  )
  expect(screen.getByRole('button', { name: 'Agent settings' }).hasAttribute('disabled')).toBe(true)
  act(() => $cronSessions.set([makeSessionInfo({ id: 'scheduled-chat', connection_id: 'jobs', profile: 'reports' })]))
  fireEvent.click(screen.getByRole('button', { name: 'Agent settings' }))
  expect(screen.getByRole('status').textContent).toBe('?agentProfile=reports&connection=jobs')
})

it('keeps a profile-only primary conversation on its source when another machine is active', () => {
  $apiRequestScope.set({ connectionId: 'foreign', profile: 'research' })
  $sessions.set([makeSessionInfo({ id: 'primary-chat', profile: 'research' })])
  $selectedStoredSessionId.set('primary-chat')
  render(
    <MemoryRouter>
      <PortalConversationHeader paneId="workspace" title="Primary conversation" />
      <Location />
    </MemoryRouter>
  )
  expect(screen.getByRole('button', { name: 'Agent settings' }).hasAttribute('disabled')).toBe(true)
  act(() => $primaryGatewayRoute.set({ connectionId: 'local', mode: 'local' }))
  fireEvent.click(screen.getByRole('button', { name: 'Agent settings' }))
  expect(screen.getByRole('status').textContent).toBe('?agentProfile=research&connection=local')
})

it('only replaces conversation chrome in Portal, leaving settings and tools alone', () => {
  function Gate({ pane }: { pane: string }) {
    return <output>{String(usePortalConversation(pane))}</output>
  }

  const view = render(<Gate pane="workspace" />)
  expect(screen.getByRole('status').textContent).toBe('true')
  act(() => $workspaceIsPage.set(true))
  expect(screen.getByRole('status').textContent).toBe('false')
  view.rerender(<Gate pane="session-tile:chat" />)
  expect(screen.getByRole('status').textContent).toBe('true')
  view.rerender(<Gate pane="terminal" />)
  expect(screen.getByRole('status').textContent).toBe('false')
  act(() => $shellChoice.set(false))
  view.rerender(<Gate pane="session-tile:chat" />)
  expect(screen.getByRole('status').textContent).toBe('false')
})

it('mounts status polling on demand and retains plugin, menu and action controls', async () => {
  const effect = vi.fn()
  const dispose = vi.fn()
  const run = vi.fn()

  function Status() {
    useEffect(() => {
      effect()

      return dispose
    }, [])

    return (
      <StatusbarControls
        items={[
          { id: 'command-center', title: 'Command Center', onSelect: run },
          { id: 'nav', label: 'Diagnostics', to: '/diagnostics' },
          { id: 'plugin', render: () => <button type="button">Plugin status</button> },
          {
            id: 'gateway',
            label: en.shell.gatewayMenu.gateway,
            variant: 'menu',
            menuItems: [{ id: 'logs', label: 'Logs', onSelect: run }]
          }
        ]}
        sidebar
      />
    )
  }

  $statusbarHiddenIds.set(['command-center'])
  render(
    <MemoryRouter>
      <PortalStatusMenu>{() => <Status />}</PortalStatusMenu>
    </MemoryRouter>
  )
  expect(effect).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'System' }))
  expect(effect).toHaveBeenCalledTimes(1)
  expect(screen.getByRole('button', { name: 'Plugin status' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Command Center' }), { shiftKey: true })
  expect(run).toHaveBeenCalledWith({ shiftKey: true })
  fireEvent.keyDown(screen.getByRole('button', { name: en.shell.gatewayMenu.gateway }), { key: 'Enter' })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Logs' }))
  expect(run).toHaveBeenCalledTimes(2)
  fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
  expect(dispose).toHaveBeenCalledTimes(1)
  fireEvent.click(screen.getByRole('button', { name: 'System' }))
  fireEvent.click(screen.getByRole('button', { name: 'Diagnostics' }))
  expect(dispose).toHaveBeenCalledTimes(2)
})

it('keeps an unlisted tile’s explicit owner alias when opening its settings', () => {
  $sessionTiles.set([
    {
      storedSessionId: 'hidden-chat',
      ownerRoute: { connectionId: 'remote', profile: 'research', targetProfile: 'default' }
    }
  ])
  render(
    <MemoryRouter>
      <PortalConversationHeader paneId="session-tile:hidden-chat" title="Unlisted" />
      <Location />
    </MemoryRouter>
  )
  fireEvent.click(screen.getByRole('button', { name: 'Agent settings' }))
  expect(screen.getByRole('status').textContent).toBe('?agentProfile=research&connection=remote')
})
