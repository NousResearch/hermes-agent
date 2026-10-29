// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { setSidebarGrouping } from '@/store/layout'
import { $newChatProfile, $profiles } from '@/store/profile'
import { $sessions } from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'

import { ChatSidebar } from './index'

const noop = () => {}

const noopAsync = async () => {}

afterEach(cleanup)

it("heads a renamed profile's owner groups with its display name, keyed by the canonical profile", () => {
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
          onResumeSession={noop}
          onRetrySessions={noopAsync}
          onTriggerCronJob={async () => {}}
        />
      </SidebarProvider>
    </MemoryRouter>
  )
  act(() => {
    $connectionsRegistry.set({
      version: 2,
      primary: 'local',
      secureTokenStorage: true,
      connections: [
        { id: 'local', label: 'This computer', kind: 'local', tokenSet: false, tokenPreview: null },
        { id: 'remote-1', label: 'Homelab', kind: 'remote', tokenSet: false, tokenPreview: null }
      ]
    })
    $profiles.set([
      { name: 'default', display_name: 'NEON', is_default: true },
      { name: 'work', is_default: false }
    ] as typeof $profiles.value)
    setSidebarGrouping('profile')
    $sessions.set(
      [
        { id: 'bare', connection_id: undefined },
        { id: 'local', connection_id: 'local' },
        { id: 'remote', connection_id: 'remote-1' }
      ].map(({ id, connection_id }) =>
        makeSessionInfo({
          id,
          connection_id,
          profile: 'default',
          title: `${id} session`,
          last_active: Date.now() / 1000
        })
      )
    )
  })

  const bare = screen.getByText('bare session').closest('[data-gateway-group]') as HTMLElement
  expect(bare.getAttribute('data-gateway-group')).toBe(JSON.stringify([null, 'default']))
  expect(within(bare).getByText('NEON')).toBeTruthy()
  fireEvent.click(within(bare).getByRole('button', { name: 'New session in NEON' }))
  expect($newChatProfile.get()).toBe('default')

  const local = screen.getByText('This computer').closest('[data-gateway-section]') as HTMLElement
  expect(within(local).getByText('NEON')).toBeTruthy()

  // The remote's own profile list was never served, so its group keeps the key
  // instead of borrowing the local profile's display name.
  const remote = screen.getByText('Homelab').closest('[data-gateway-section]') as HTMLElement
  expect(within(remote).getByText('default')).toBeTruthy()
  expect(within(remote).queryByText('NEON')).toBeNull()
})
