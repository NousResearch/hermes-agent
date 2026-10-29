// @vitest-environment jsdom
import { act, cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $sidebarRowMeta, setSidebarGrouping } from '@/store/layout'
import { $profiles, $showAllProfiles } from '@/store/profile'
import { $sessionProfilesUsage, $sessions } from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'

import { ChatSidebar } from './index'

const noop = () => {}

const noopAsync = async () => {}

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
          onResumeSession={noop}
          onRetrySessions={noopAsync}
          onTriggerCronJob={async () => {}}
        />
      </SidebarProvider>
    </MemoryRouter>
  )

afterEach(() => {
  cleanup()
  delete window.document.documentElement.dataset.hermesDesktopHost
  $showAllProfiles.set(false)
  $sidebarRowMeta.set([])
  $sessionProfilesUsage.set({})
  $sessions.set([])
})

// The Webapp publishes no connection registry: its one gateway is the `local` source.
it('names the Webapp gateway section after the host instead of its raw `local` id', () => {
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
  mount()
  act(() => {
    $connectionsRegistry.set(null)
    $profiles.set([
      { name: 'default', is_default: true },
      { name: 'work', is_default: false }
    ] as typeof $profiles.value)
    $showAllProfiles.set(true)
    setSidebarGrouping('profile')
    $sidebarRowMeta.set(['cost', 'tokens'])
    $sessionProfilesUsage.set({ default: { cost_usd: 3, tokens: 100 }, work: { cost_usd: 1.25, tokens: 40 } })
    $sessions.set(
      [
        ['listed', 'default'],
        ['new-chat', 'default'],
        ['work-chat', 'work']
      ].map(([id, profile]) =>
        makeSessionInfo({ id, connection_id: 'local', profile, title: `${id} title`, last_active: Date.now() / 1000 })
      )
    )
  })

  const sections = window.document.querySelectorAll('[data-gateway-section]')
  expect(sections).toHaveLength(1)
  const gateway = sections[0] as HTMLElement
  expect(within(gateway).getByText('Hermes host')).toBeTruthy()
  expect(screen.queryByText('local')).toBeNull()
  expect(gateway.querySelectorAll('[data-gateway-group]')).toHaveLength(2)
  expect(within(gateway).getByText('listed title')).toBeTruthy()
  expect(within(gateway).getByText('new-chat title')).toBeTruthy()
  expect(within(gateway).getByText('work-chat title')).toBeTruthy()

  // The serving host's per-profile totals belong to its `local` profile groups.
  const groupOf = (text: RegExp) =>
    screen.getByText(text).closest('[data-gateway-group]')?.getAttribute('data-gateway-group')

  expect(groupOf(/\$3\.00/)).toBe(JSON.stringify(['local', 'default']))
  expect(groupOf(/\$1\.25/)).toBe(JSON.stringify(['local', 'work']))
})

// A profile with no chats still gets its own section, sourced from the profile list, on both hosts.
it.each([
  { host: 'Webapp', browser: true, connectionId: 'local' },
  { host: 'native Desktop', browser: false, connectionId: null }
])('shows a profile with no chats as its own section ($host)', ({ browser, connectionId }) => {
  if (browser) {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
  }

  mount()
  act(() => {
    $connectionsRegistry.set(null)
    $profiles.set([
      { name: 'default', is_default: true },
      { name: 'quiet', is_default: false }
    ] as typeof $profiles.value)
    $showAllProfiles.set(true)
    setSidebarGrouping('profile')
    $sessions.set([
      makeSessionInfo({
        id: 'one',
        ...(connectionId ? { connection_id: connectionId } : {}),
        profile: 'default',
        title: 'only chat',
        last_active: Date.now() / 1000
      })
    ])
  })

  const quiet = window.document.querySelector(
    `[data-gateway-group='${JSON.stringify([connectionId, 'quiet'])}']`
  ) as HTMLElement | null

  expect(quiet).toBeTruthy()
  expect(within(quiet!).getByText('quiet')).toBeTruthy()
  expect(within(quiet!).getByText('No sessions yet')).toBeTruthy()
  expect(window.document.querySelectorAll('[data-gateway-group-empty]')).toHaveLength(1)
  expect(within(quiet!).getByRole('button', { name: 'New session in quiet' })).toBeTruthy()
  expect(screen.getByText('only chat')).toBeTruthy()
})
