import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $fleetRoster } from '@/store/fleet-roster'
import { $pinnedSessionIds, $sidebarPinsOpen, $sidebarRecentsOpen } from '@/store/layout'
import { $onboardingDecided, $onboardingSurfaces } from '@/store/onboarding-presence'
import type * as ProfileModule from '@/store/profile'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $profiles,
  $profileScope,
  $showAllProfiles,
  ALL_PROFILES
} from '@/store/profile'
import {
  $cronSessions,
  $gatewayState,
  $messagingPlatformTotals,
  $messagingSessions,
  $messagingTruncated,
  $sessionProfilesTruncated,
  $sessions,
  $sessionsLoading
} from '@/store/session'
import { $settingsScopeProfile } from '@/store/settings-scope'
import { $agentSidebarChoice, setSimplifiedAgentMode } from '@/store/simplified-agent-mode'
import { makeSessionInfo } from '@/test/session-info'

import { SimplifiedAgentOffer } from './simplified-agent-offer'
import { SimplifiedAgentSidebar } from './simplified-agent-sidebar'

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<typeof ProfileModule>()),
  ensureGatewayProfile: vi.fn(async () => {})
}))
const navigate = vi.fn()
const resume = vi.fn()

const noop = () => {}

function Route() {
  return <output data-testid="route">{useLocation().search}</output>
}

function mount(overrides: Partial<ComponentProps<typeof SimplifiedAgentSidebar>> = {}) {
  return render(
    <MemoryRouter>
      <SidebarProvider>
        <SimplifiedAgentSidebar
          currentView="chat"
          onArchiveSession={noop}
          onBranchSession={noop}
          onDeleteSession={noop}
          onLoadMoreSessions={noop}
          onManageCronJob={noop}
          onNavigate={navigate}
          onNewSessionInWorkspace={noop}
          onNewSessionSplit={noop}
          onResumeSession={resume}
          onRetrySessions={noop}
          onTriggerCronJob={async () => {}}
          {...overrides}
        />
        <Route />
      </SidebarProvider>
    </MemoryRouter>
  )
}

beforeEach(() => {
  setSimplifiedAgentMode(true)
  $connectionsRegistry.set(null)
  $fleetRoster.set(null)
  $profiles.set([
    { name: 'default', is_default: true },
    { name: 'research', display_name: 'Research', is_default: false }
  ] as typeof $profiles.value)
  $sessions.set([
    makeSessionInfo({ id: 'pinned', title: 'Pinned plan', profile: 'research', pinned: true }),
    makeSessionInfo({ id: 'regular', title: 'Regular chat', profile: 'default' })
  ])
  $sessionsLoading.set(false)
  $messagingTruncated.set(false)
  $messagingPlatformTotals.set({})
  $sessionProfilesTruncated.set({})
  $messagingSessions.set([])
  $cronSessions.set([])
  $pinnedSessionIds.set([])
  $sidebarPinsOpen.set(true)
  $sidebarRecentsOpen.set(true)
  navigate.mockClear()
  resume.mockClear()
})
afterEach(() => {
  cleanup()
  setSimplifiedAgentMode(false)
  $sessions.set([])
  $fleetRoster.set(null)
})
it('keeps server-pinned chats reachable and resumes the exact session; New session uses the usual action', () => {
  mount()
  expect(screen.getByText('Pinned plan')).toBeTruthy()

  const agentGroup = screen
    .getByRole('button', { name: /Agent settings: default/ })
    .closest('[role=group]') as HTMLElement

  expect(within(agentGroup).getByText('Regular chat')).toBeTruthy()
  expect(within(agentGroup).queryByText('Pinned plan')).toBeNull()
  expect(screen.queryByText('Chats', { exact: true })).toBeNull()
  expect(screen.getAllByLabelText('Agents: default').length).toBeGreaterThan(0)
  fireEvent.click(screen.getByText('Regular chat'))
  expect(resume).toHaveBeenCalledWith('regular', expect.objectContaining({ profile: 'default' }))
  $newChatProfile.set('research')
  fireEvent.click(screen.getByRole('button', { name: /New session/i }))
  expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ action: 'new-session', id: 'new-session' }))
  expect($newChatProfile.get()).toBeNull()
})
it('opens the clicked agent settings, including agents with no chats', async () => {
  mount()
  fireEvent.click(screen.getByRole('button', { name: /Agent settings: Research/ }))
  await waitFor(() => expect($settingsScopeProfile.get()).toBe('research'))
  await waitFor(() => expect(screen.getByTestId('route').textContent).toBe('?agent=research'))
})
it('offers the mode once after onboarding and persists both yes and no without changing the classic browse preference', () => {
  $agentSidebarChoice.set(null)
  $showAllProfiles.set(false)
  $activeGatewayProfile.set('research')
  $gatewayState.set('open')
  $onboardingDecided.set(false)
  $onboardingSurfaces.set(new Set())
  const view = render(<SimplifiedAgentOffer />)
  expect(screen.queryByRole('dialog')).toBeNull()
  act(() => $onboardingDecided.set(true))
  fireEvent.click(screen.getByRole('button', { name: 'No, keep current sidebar' }))
  expect($agentSidebarChoice.get()).toBe(false)
  view.unmount()
  render(<SimplifiedAgentOffer />)
  expect(screen.queryByRole('dialog')).toBeNull()
  act(() => $agentSidebarChoice.set(null))
  fireEvent.click(screen.getByRole('button', { name: 'Yes, try it' }))
  expect($profileScope.get()).toBe(ALL_PROFILES)
  expect($showAllProfiles.get()).toBe(false)
  expect(localStorage.getItem('hermes.desktop.simplifiedAgentMode.v1')).toBe('true')
  act(() => setSimplifiedAgentMode(false))
  expect($profileScope.get()).toBe('research')
})

it('keeps same-named agents on different gateways separate and groups them by gateway', () => {
  $profiles.set([])
  $sessions.set([
    makeSessionInfo({ id: 'alpha', title: 'Alpha chat', profile: 'research', connection_id: 'a' }),
    makeSessionInfo({ id: 'beta', title: 'Beta chat', profile: 'research', connection_id: 'b' }),
    makeSessionInfo({ id: 'orphan', title: 'Unlisted agent chat', profile: 'writer', connection_id: 'a' })
  ])
  $fleetRoster.set({
    sources: [
      { connectionId: 'a', kind: 'remote', label: 'Alpha', reachable: true },
      { connectionId: 'b', kind: 'remote', label: 'Beta', reachable: true }
    ],
    agents: [
      {
        connectionId: 'b',
        connectionKind: 'remote',
        connectionLabel: 'Beta',
        profile: 'research',
        handle: 'research-beta'
      },
      {
        connectionId: 'a',
        connectionKind: 'remote',
        connectionLabel: 'Alpha',
        profile: 'research',
        handle: 'research-alpha'
      }
    ]
  })
  mount()
  const agents = screen.getAllByRole('button', { name: /Agent settings: research/ })
  expect(agents).toHaveLength(2)
  expect(agents[0].getAttribute('aria-label')).toContain('Alpha')
  expect(agents[1].getAttribute('aria-label')).toContain('Beta')
  const alpha = within(agents[0].closest('[role=group]') as HTMLElement)
  const beta = within(agents[1].closest('[role=group]') as HTMLElement)
  expect(alpha.getByText('Alpha chat')).toBeTruthy()
  expect(alpha.queryByText('Beta chat')).toBeNull()
  expect(beta.getByText('Beta chat')).toBeTruthy()
  expect(screen.getByText('Unlisted agent chat')).toBeTruthy()
})

it('retains same-id chats across profiles and gateways, including pinned twins', () => {
  $sessions.set([
    makeSessionInfo({ id: 'same', title: 'Default twin', profile: 'default' }),
    makeSessionInfo({ id: 'same', title: 'Research twin', profile: 'research', pinned: true }),
    makeSessionInfo({ id: 'same', title: 'Remote twin', profile: 'research', connection_id: 'remote', pinned: true })
  ])
  mount()

  for (const title of ['Default twin', 'Research twin', 'Remote twin']) {
    fireEvent.click(screen.getByText(title))
  }

  expect(resume.mock.calls.map(([, row]) => [row.profile, row.connection_id])).toEqual([
    ['default', undefined],
    ['research', undefined],
    ['research', 'remote']
  ])
})

it('loads older messaging chats when recents are exhausted', async () => {
  $messagingTruncated.set(true)
  $messagingPlatformTotals.set({ telegram: 50 })
  const loadMessaging = vi.fn()
  mount({ onLoadMoreMessaging: loadMessaging })
  fireEvent.click(screen.getByRole('button', { name: 'Load more' }))
  await waitFor(() => expect(loadMessaging).toHaveBeenCalledWith('telegram'))
})
