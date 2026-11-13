// @vitest-environment jsdom
import { act, cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { setSidebarGrouping } from '@/store/layout'
import { $profiles } from '@/store/profile'
import { $sessions } from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'

import { ChatSidebar } from './index'

const noop = () => {}

const noopAsync = async () => {}
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
          onRetrySessions={noopAsync}
          onTriggerCronJob={async () => {}}
        />
      </SidebarProvider>
    </MemoryRouter>
  )

afterEach(cleanup)

// The primary's bare pool ([null, p]) and rows naming the primary ([primary, p]) are one gateway:
// every primary profile sits once inside the primary's section, never as a headerless root.
it('files bare primary rows and empty primary profiles in the primary section, one group per profile', () => {
  mount()
  act(() => {
    $connectionsRegistry.set({
      version: 2,
      primary: 'local',
      secureTokenStorage: true,
      connections: [
        { id: 'local', label: 'This device', kind: 'local', tokenSet: false, tokenPreview: null },
        { id: 'remote-1', label: 'Homelab', kind: 'remote', tokenSet: false, tokenPreview: null }
      ]
    })
    $profiles.set([
      { name: 'default', is_default: true },
      { name: 'research', is_default: false },
      { name: 'writer', is_default: false }
    ] as typeof $profiles.value)
    setSidebarGrouping('profile')
    const now = Date.now() / 1000
    $sessions.set([
      makeSessionInfo({
        id: 'lab',
        connection_id: 'remote-1',
        profile: 'default',
        title: 'lab chat',
        last_active: now
      }),
      makeSessionInfo({ id: 'bare-default', profile: 'default', title: 'bare default', last_active: now - 1 }),
      makeSessionInfo({ id: 'bare-research', profile: 'research', title: 'bare research', last_active: now - 2 }),
      makeSessionInfo({
        id: 'writer',
        connection_id: 'local',
        profile: 'writer',
        title: 'writer chat',
        last_active: now - 3
      }),
      makeSessionInfo({ id: 'bare-writer', profile: 'writer', title: 'bare writer', last_active: now - 4 })
    ])
  })

  const device = screen.getByText('This device').closest('[data-gateway-section]') as HTMLElement
  const homelab = screen.getByText('Homelab').closest('[data-gateway-section]') as HTMLElement

  // No profile group floats outside a gateway section.
  for (const group of window.document.querySelectorAll('[data-gateway-group]')) {
    expect(group.parentElement?.closest('[data-gateway-section]')).toBeTruthy()
  }

  expect(within(device).getAllByText('writer')).toHaveLength(1)
  expect(within(device).getByText('writer chat')).toBeTruthy()
  expect(within(device).getByText('bare writer')).toBeTruthy()
  expect(within(device).getByText('bare research')).toBeTruthy()
  expect(within(device).getByText('bare default')).toBeTruthy()
  expect(within(device).queryByText('lab chat')).toBeNull()
  expect(within(homelab).getByText('lab chat')).toBeTruthy()
  expect(device.querySelectorAll('[data-gateway-group]')).toHaveLength(3)
})
