import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import * as sessionsApi from '@/api/sessions'
import { SidebarProvider } from '@/components/ui/sidebar'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $fleetRoster } from '@/store/fleet-roster'
import { $pinnedSessionIds, $sidebarPinsOpen, $sidebarRecentsOpen } from '@/store/layout'
import { $onboardingDecided, $onboardingSurfaces } from '@/store/onboarding-presence'
import type * as ProfileModule from '@/store/profile'
import {
  $activeGatewayProfile,
  $profiles,
  $profileScope,
  $showAllProfiles,
  ALL_PROFILES,
  newSessionInProfile
} from '@/store/profile'
import {
  $cronSessions,
  $gatewayState,
  $messagingPlatformTotals,
  $messagingSessions,
  $messagingTruncated,
  $selectedStoredSessionId,
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
  ensureGatewayProfile: vi.fn(async () => {}),
  newSessionInProfile: vi.fn()
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
  $apiRequestScope.set({ connectionId: null, profile: null })
  $gatewayState.set('idle')
  $selectedStoredSessionId.set(null)
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
  $apiRequestScope.set({ connectionId: null, profile: null })
  $gatewayState.set('idle')
  $selectedStoredSessionId.set(null)
  $fleetRoster.set(null)
})
it('keeps server-pinned chats reachable and resumes the exact session', () => {
  mount()
  expect(screen.getAllByText('Pinned plan')).toHaveLength(2)

  const research = screen
    .getByRole('button', { name: /Agent settings: Research/ })
    .closest('[role=group]') as HTMLElement

  fireEvent.click(within(research).getByText('Pinned plan'))
  expect(resume).toHaveBeenCalledWith('pinned', expect.objectContaining({ profile: 'research' }))

  const agentGroup = screen
    .getByRole('button', { name: /Agent settings: default/ })
    .closest('[role=group]') as HTMLElement

  expect(within(agentGroup).getByText('Regular chat')).toBeTruthy()
  expect(within(agentGroup).queryByText('Pinned plan')).toBeNull()
  expect(screen.queryByText('Chats', { exact: true })).toBeNull()
  expect(within(agentGroup).queryByLabelText('Agents: default')).toBeNull()
  expect(screen.getByLabelText('Agents: research')).toBeTruthy()
  fireEvent.click(screen.getByText('Regular chat'))
  expect(resume).toHaveBeenCalledWith('regular', expect.objectContaining({ profile: 'default' }))
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
    fireEvent.click(screen.getAllByText(title)[0])
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

it('starts a chat directly from an agent row without opening its settings', () => {
  mount()
  fireEvent.click(screen.getByRole('button', { name: 'New session in Research' }))
  expect(newSessionInProfile).toHaveBeenCalledWith('research')
  expect(screen.getByTestId('route').textContent).toBe('')
})

it('offers Unpin under the agent as well as in Pinned, keeping the owning profile', async () => {
  const update = vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({})

  try {
    mount()

    const research = screen
      .getByRole('button', { name: /Agent settings: Research/ })
      .closest('[role=group]') as HTMLElement

    fireEvent.keyDown(within(research).getByRole('button', { name: 'Session actions' }), { key: 'Enter' })
    fireEvent.click(await screen.findByRole('menuitem', { name: /Unpin/ }))
    expect(screen.getAllByText('Pinned plan')).toHaveLength(1)
    expect(within(research).getByText('Pinned plan')).toBeTruthy()
    await waitFor(() =>
      expect(update).toHaveBeenCalledWith(
        'pinned',
        'PATCH',
        { pinned: false },
        expect.objectContaining({ profile: 'research' })
      )
    )
  } finally {
    update.mockRestore()
  }
})

it('shows last activity under agents, keeps profile labels on pins, and refreshes age while idle', () => {
  vi.useFakeTimers()
  const now = new Date('2026-11-03T12:00:00Z').getTime()
  vi.setSystemTime(now)

  try {
    $sessions.set([
      makeSessionInfo({
        id: 'recent',
        title: 'Recent work',
        profile: 'research',
        pinned: true,
        started_at: now / 1000 - 3 * 86400,
        last_active: now / 1000 - 5 * 3600
      }),
      makeSessionInfo({
        id: 'older',
        title: 'Older work',
        profile: 'research',
        started_at: now / 1000 - 86400,
        last_active: 0
      })
    ])
    mount()

    const research = screen
      .getByRole('button', { name: /Agent settings: Research/ })
      .closest('[role=group]') as HTMLElement

    expect(within(research).getByText('5h')).toBeTruthy()
    expect(within(research).getByText('1d')).toBeTruthy()
    expect(within(research).queryByLabelText('Agents: research')).toBeNull()
    expect(screen.getByLabelText('Agents: research')).toBeTruthy()
    expect(research.querySelector('time')?.getAttribute('datetime')).toBe('2026-11-03T07:00:00.000Z')
    act(() => {
      vi.advanceTimersByTime(3600000)
    })
    expect(within(research).getByText('6h')).toBeTruthy()
  } finally {
    cleanup()
    vi.useRealTimers()
  }
})

it('marks only the active gateway and updates its status when the live connection drops', () => {
  $apiRequestScope.set({ connectionId: 'a', profile: 'default' })
  $gatewayState.set('open')
  $connectionsRegistry.set({
    version: 1,
    secureTokenStorage: true,
    primary: 'a',
    connections: [
      {
        id: 'a',
        kind: 'remote',
        label: 'Alpha',
        url: 'https://user:secret@alpha.example:9137/private?token=hidden',
        tokenSet: false,
        tokenPreview: null
      }
    ]
  })
  $fleetRoster.set({
    sources: [
      { connectionId: 'a', kind: 'remote', label: 'Alpha', reachable: true },
      { connectionId: 'b', kind: 'remote', label: 'Beta', reachable: false }
    ],
    agents: [
      { connectionId: 'a', connectionKind: 'remote', connectionLabel: 'Alpha', profile: 'default', handle: 'a' },
      { connectionId: 'b', connectionKind: 'remote', connectionLabel: 'Beta', profile: 'default', handle: 'b' }
    ]
  })
  mount()
  const alpha = screen.getByRole('group', { name: 'Alpha' })
  const beta = screen.getByRole('group', { name: 'Beta' })
  expect(within(alpha).getByText('alpha.example:9137')).toBeTruthy()
  expect(alpha.textContent).not.toContain('secret')
  expect(alpha.textContent).not.toContain('hidden')
  expect(alpha.getAttribute('aria-current')).toBe('true')
  expect(beta.hasAttribute('aria-current')).toBe(false)
  expect(within(alpha).getByText('Connected')).toBeTruthy()
  expect(within(beta).getByText('Offline')).toBeTruthy()
  act(() => $gatewayState.set('closed'))
  expect(within(alpha).queryByText('Connected')).toBeNull()
  expect(within(alpha).getByText('Offline')).toBeTruthy()
  act(() => $gatewayState.set('connecting'))
  expect(within(alpha).getByText('Connecting')).toBeTruthy()
  const alphaToggle = within(alpha).getByRole('button', { name: 'Alpha' })
  const betaToggle = within(beta).getByRole('button', { name: 'Beta' })
  fireEvent.click(alphaToggle)
  fireEvent.click(betaToggle)
  fireEvent.click(screen.getByText('Pinned plan'))
  expect(alphaToggle.getAttribute('aria-expanded')).toBe('true')
  expect(betaToggle.getAttribute('aria-expanded')).toBe('false')
  expect(screen.getAllByText('Pinned plan')).toHaveLength(2)
})

it('preserves manual agent expansion when selecting chats and only reveals a pinned chat owner', () => {
  mount()
  const research = screen.getByRole('button', { name: 'Research', expanded: true })
  const defaultAgent = screen.getByRole('button', { name: 'default', expanded: true })
  fireEvent.click(research)
  fireEvent.click(screen.getByText('Regular chat'))
  act(() => {
    $selectedStoredSessionId.set('regular')
    $activeGatewayProfile.set('default')
  })
  expect(research.getAttribute('aria-expanded')).toBe('false')
  expect(defaultAgent.getAttribute('aria-expanded')).toBe('true')
  fireEvent.click(research)
  act(() => $selectedStoredSessionId.set('another-chat'))
  expect(research.getAttribute('aria-expanded')).toBe('true')
  fireEvent.click(research)
  fireEvent.click(defaultAgent)
  fireEvent.click(screen.getByText('Pinned plan'))
  act(() => {
    $selectedStoredSessionId.set('pinned')
    $activeGatewayProfile.set('research')
  })
  expect(research.getAttribute('aria-expanded')).toBe('true')
  expect(defaultAgent.getAttribute('aria-expanded')).toBe('false')
  expect(screen.getAllByText('Pinned plan')).toHaveLength(2)
  fireEvent.click(research)
  fireEvent.click(screen.getByText('Pinned plan'))
  expect(research.getAttribute('aria-expanded')).toBe('true')
})
