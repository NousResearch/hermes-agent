// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $sidebarRowMeta, setSidebarGrouping } from '@/store/layout'
import { $newChatRoute, $profiles } from '@/store/profile'
import { $sessionProfilesUsage, $sessions } from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'

import { ChatSidebar } from './index'

const noop = () => {}
const resume = vi.fn()

const mount = () =>
  render(
    <MemoryRouter>
      <SidebarProvider>
        <ChatSidebar
          currentView="chat"
          onArchiveSession={noop}
          onBranchSession={noop}
          onDeleteSession={noop}
          onLoadMoreSessions={noop}
          onManageCronJob={noop}
          onNavigate={noop}
          onNewSessionInWorkspace={noop}
          onNewSessionSplit={noop}
          onResumeSession={resume}
          onTriggerCronJob={async () => {}}
        />
      </SidebarProvider>
    </MemoryRouter>
  )

afterEach(cleanup)

it('keeps equal profile names on separate gateways and routes section creation and row navigation to their owner', () => {
  mount()
  act(() => {
    $connectionsRegistry.set({
      version: 2,
      primary: 'local',
      secureTokenStorage: true,
      connections: [
        { id: 'local', label: 'This computer', kind: 'local', tokenSet: false, tokenPreview: null },
        { id: 'remote-1', label: 'Homelab', kind: 'remote', tokenSet: false, tokenPreview: null },
        { id: 'cloud-1', label: 'Cloud workspace', kind: 'cloud', tokenSet: false, tokenPreview: null }
      ]
    })
    $profiles.set([
      { name: 'default', is_default: true },
      { name: 'work', is_default: false }
    ] as typeof $profiles.value)
    setSidebarGrouping('profile')
    $sidebarRowMeta.set(['cost', 'tokens'])
    $sessionProfilesUsage.set({ default: { cost_usd: 3, tokens: 100 } })
    $sessions.set(
      ['local', 'remote-1', 'cloud-1'].map(connection_id =>
        makeSessionInfo({
          id: connection_id,
          connection_id,
          profile: 'default',
          title: `${connection_id} session`,
          last_active: Date.now() / 1000
        })
      )
    )
  })
  act(() =>
    $sessions.set([
      ...$sessions.get(),
      makeSessionInfo({ id: 'legacy', profile: 'default', title: 'Legacy session', last_active: Date.now() / 1000 })
    ])
  )
  expect(screen.getAllByText(/\$3\.00/)).toHaveLength(1)
  expect(
    screen
      .getByText(/\$3\.00/)
      .closest('[data-gateway-group]')
      ?.getAttribute('data-gateway-group')
  ).toBe(JSON.stringify([null, 'default']))
  expect(screen.getByText('This computer · default')).toBeTruthy()
  expect(screen.getByText('Homelab · default')).toBeTruthy()
  expect(screen.getByText('Cloud workspace · default')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'New session in Homelab · default' }))
  expect($newChatRoute.get()).toMatchObject({ connectionId: 'remote-1', profile: 'default' })
  fireEvent.click(screen.getByText('cloud-1 session'))
  expect(resume).toHaveBeenLastCalledWith(
    'cloud-1',
    expect.objectContaining({ connection_id: 'cloud-1', profile: 'default' })
  )
  const group = screen.getByText('Homelab · default').closest('[data-gateway-group]')!
  fireEvent.click(within(group as HTMLElement).getByRole('button', { name: 'Hide Homelab · default sessions' }))
  expect(screen.queryByText('remote-1 session')).toBeNull()
  expect(screen.getByText('local session')).toBeTruthy()
})
