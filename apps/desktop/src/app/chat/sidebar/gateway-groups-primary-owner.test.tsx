// @vitest-environment jsdom
import { act, cleanup, render, within } from '@testing-library/react'
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
  $connectionsRegistry.set(null)
  $showAllProfiles.set(false)
  $sidebarRowMeta.set([])
  $sessionProfilesUsage.set({})
  $sessions.set([])
})

// Native Desktop with a registry: the primary `local` pool lists its rows untagged, while
// optimistic rows / cache patches / fleet-rail new chats name `local` exactly. Both are the same
// gateway, so they must land in one section with each profile exactly once.
it('nests bare and `local`-tagged primary rows plus empty profiles under one primary section', () => {
  mount()
  act(() => {
    $connectionsRegistry.set({
      version: 2,
      primary: 'local',
      secureTokenStorage: true,
      connections: [
        { id: 'local', label: 'This device', kind: 'local', tokenSet: false, tokenPreview: null },
        { id: 'hermes-watchtower-2', label: 'Watchtower', kind: 'remote', tokenSet: false, tokenPreview: null }
      ]
    })
    $profiles.set([
      { name: 'default', is_default: true },
      { name: 'coder', is_default: false },
      { name: 'tranquillity-sales', is_default: false },
      { name: 'quiet', is_default: false }
    ] as typeof $profiles.value)
    $showAllProfiles.set(true)
    setSidebarGrouping('profile')
    $sidebarRowMeta.set(['cost', 'tokens'])
    $sessionProfilesUsage.set({ coder: { cost_usd: 2, tokens: 50 } })
    const now = Date.now() / 1000
    $sessions.set([
      makeSessionInfo({ id: 'sales-bare', profile: 'tranquillity-sales', title: 'sales bare', last_active: now }),
      makeSessionInfo({
        id: 'sales-tagged',
        connection_id: 'local',
        profile: 'tranquillity-sales',
        title: 'sales tagged',
        last_active: now
      }),
      makeSessionInfo({ id: 'coder-bare', profile: 'coder', title: 'coder bare', last_active: now }),
      makeSessionInfo({ id: 'default-bare', profile: 'default', title: 'default bare', last_active: now }),
      makeSessionInfo({
        id: 'tower',
        connection_id: 'hermes-watchtower-2',
        profile: 'tranquillity-sales',
        title: 'tower chat',
        last_active: now
      })
    ])
  })

  const sections = [...window.document.querySelectorAll('[data-gateway-section]')] as HTMLElement[]
  expect(sections.map(section => section.getAttribute('data-gateway-section')).sort()).toEqual(
    [JSON.stringify(['gateway', 'hermes-watchtower-2']), JSON.stringify(['gateway', 'local'])].sort()
  )

  // Nothing floats at top level: every profile group sits inside a gateway section.
  for (const group of window.document.querySelectorAll('[data-gateway-group]')) {
    expect(group.parentElement?.closest('[data-gateway-section]')).toBeTruthy()
  }

  const primary = sections.find(s => s.getAttribute('data-gateway-section') === JSON.stringify(['gateway', 'local']))!

  const primaryGroups = [...primary.querySelectorAll('[data-gateway-group]')].map(g =>
    g.getAttribute('data-gateway-group')
  )

  expect(primaryGroups.sort()).toEqual(
    ['coder', 'default', 'quiet', 'tranquillity-sales'].map(profile => JSON.stringify(['local', profile])).sort()
  )

  const sales = primary.querySelector(
    `[data-gateway-group='${JSON.stringify(['local', 'tranquillity-sales'])}']`
  ) as HTMLElement

  expect(within(sales).getByText('sales bare')).toBeTruthy()
  expect(within(sales).getByText('sales tagged')).toBeTruthy()
  expect(within(primary).queryByText('tower chat')).toBeNull()

  // The primary pool's profile-keyed totals still show on its own groups (and only once).
  expect(within(primary).getAllByText(/\$2\.00/)).toHaveLength(1)

  const tower = sections.find(s => s !== primary)!
  expect([...tower.querySelectorAll('[data-gateway-group]')].map(g => g.getAttribute('data-gateway-group'))).toEqual([
    JSON.stringify(['hermes-watchtower-2', 'tranquillity-sales'])
  ])
  expect(within(tower).getByText('tower chat')).toBeTruthy()
  expect(within(tower).queryByText(/\$2\.00/)).toBeNull()
})

// No registry (single-gateway native install): flat profile groups, no gateway sections.
it('keeps flat profile groups when no connections registry exists', () => {
  mount()
  act(() => {
    $connectionsRegistry.set(null)
    $profiles.set([{ name: 'default', is_default: true }] as typeof $profiles.value)
    setSidebarGrouping('profile')
    $sessions.set([
      makeSessionInfo({ id: 'x', profile: 'default', title: 'flat chat', last_active: Date.now() / 1000 })
    ])
  })

  expect(window.document.querySelectorAll('[data-gateway-section]')).toHaveLength(0)
  expect(window.document.querySelector(`[data-gateway-group='${JSON.stringify([null, 'default'])}']`)).toBeTruthy()
})
