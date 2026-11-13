import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor, within } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import * as apiClient from '@/api/client'
import * as sessionsApi from '@/api/sessions'
import { useSessionListActions } from '@/app/session/hooks/use-session-list-actions'
import { StableNavigateProvider } from '@/app/stable-navigate'
import { SidebarProvider } from '@/components/ui/sidebar'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { $openBotChat, $pendingBotOpen } from '@/plugins/hermes-bots/bot-state'
import { openRosterBot } from '@/plugins/hermes-bots/roster-actions'
import {
  $hiddenSidebarAgents,
  $hiddenSidebarAgentsExpanded,
  $portalAgentsSectionOpen,
  $portalPinnedSectionOpen,
  $sidebarAgentExpanded,
  $sidebarAgentScroll,
  agentSidebarKey,
  setSidebarAgentHidden
} from '@/store/agent-sidebar-preferences'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $fleetRoster } from '@/store/fleet-roster'
import {
  $pinnedSessionIds,
  $sidebarOpen,
  $sidebarPinsOpen,
  $sidebarRecentsOpen,
  requestSessionSearchFocus,
  setSidebarOpen
} from '@/store/layout'
import type * as ProfileModule from '@/store/profile'
import {
  $activeGatewayProfile,
  $profiles,
  $profileScope,
  $sessionListProfileScope,
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
  $sessionsLoadError,
  $sessionsLoading
} from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import { $settingsScopeProfile } from '@/store/settings-scope'
import { setPortalPreferred } from '@/store/shell-variant'
import { deferred } from '@/test/deferred'
import { makeSessionInfo } from '@/test/session-info'

import { PortalSidebar } from './portal/portal-sidebar'
import { $simplifiedPinnedRows } from './use-simplified-pin-writes'

vi.mock('@/plugins/hermes-bots/roster-actions', () => ({ openRosterBot: vi.fn(async () => true) }))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<typeof ProfileModule>()),
  ensureGatewayProfile: vi.fn(async () => {}),
  newSessionInProfile: vi.fn()
}))
describe('Portal sidebar', () => {
  const Sidebar = PortalSidebar
  const navigate = vi.fn()
  const resume = vi.fn()

  const noop = () => {}

  function Route() {
    return <output data-testid="route">{useLocation().search}</output>
  }

  function mount(overrides: Partial<ComponentProps<typeof PortalSidebar>> = {}) {
    return render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MemoryRouter><StableNavigateProvider>
          <RootTooltipProvider>
            <SidebarProvider>
              <Sidebar
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
          </RootTooltipProvider>
        </StableNavigateProvider></MemoryRouter>
      </QueryClientProvider>
    )
  }

  /** Types into session search, bringing the field out from the header's search button first. */
  function searchFor(value: string) {
    if (!screen.queryByRole('textbox', { name: 'Search sessions' })) {
      fireEvent.click(screen.getByRole('button', { name: 'Search sessions' }))
    }

    fireEvent.change(screen.getByRole('textbox', { name: 'Search sessions' }), { target: { value } })
  }

  beforeEach(() => {
    $hiddenSidebarAgents.set([])
    $hiddenSidebarAgentsExpanded.set(false)
    $sidebarAgentExpanded.set({})
    $sidebarAgentScroll.set(0)
    setPortalPreferred(true)
    $apiRequestScope.set({ connectionId: null, profile: null })
    $gatewayState.set('idle')
    $selectedStoredSessionId.set(null)
    $sessionTiles.set([])
    $openBotChat.set(null)
    $pendingBotOpen.set(null)
    vi.mocked(openRosterBot).mockClear()
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
    $sessionsLoadError.set(false)
    $messagingTruncated.set(false)
    $messagingPlatformTotals.set({})
    $sessionProfilesTruncated.set({})
    $messagingSessions.set([])
    $cronSessions.set([])
    $pinnedSessionIds.set([])
    $sidebarPinsOpen.set(true)
    $sidebarRecentsOpen.set(true)
    $portalAgentsSectionOpen.set(true)
    $portalPinnedSectionOpen.set(true)
    navigate.mockClear()
    resume.mockClear()
  })
  afterEach(async () => {
    cleanup()
    // Radix restores focus on the next task; finish before JSDOM is torn down.
    const { promise, resolve } = Promise.withResolvers<void>()
    setTimeout(resolve, 0)
    await promise
    $simplifiedPinnedRows.set(new Map())
    vi.restoreAllMocks()
    setPortalPreferred(false)
    $sessions.set([])
    $sessionTiles.set([])
    $apiRequestScope.set({ connectionId: null, profile: null })
    $gatewayState.set('idle')
    $selectedStoredSessionId.set(null)
    $openBotChat.set(null)
    $pendingBotOpen.set(null)
    $fleetRoster.set(null)
  })
  it('focuses session search after reopening and remounting the redesigned sidebar', async () => {
    for (const initiallyOpen of [true, false, true]) {
      act(() => setSidebarOpen(initiallyOpen))
      const sidebar = mount()
      act(() => requestSessionSearchFocus())
      await waitFor(() => expect(document.activeElement).toBe(screen.getByRole('textbox')))
      expect($sidebarOpen.get()).toBe(true)
      sidebar.unmount()
    }
  })

  it('keeps search behind the header button: it reveals and focuses the field, and Escape or the button puts it away', async () => {
    mount()
    const toggle = screen.getByRole('button', { name: 'Search sessions' })
    expect(screen.queryByRole('textbox', { name: 'Search sessions' })).toBeNull()
    expect(toggle.getAttribute('aria-expanded')).toBe('false')

    fireEvent.click(toggle)
    const field = screen.getByRole('textbox', { name: 'Search sessions' })
    await waitFor(() => expect(document.activeElement).toBe(field))
    expect(toggle.getAttribute('aria-expanded')).toBe('true')

    fireEvent.keyDown(field, { key: 'Escape' })
    expect(screen.queryByRole('textbox', { name: 'Search sessions' })).toBeNull()
    expect(document.activeElement).toBe(toggle)

    // Putting the search away ends it: it comes back empty.
    searchFor('missing')
    fireEvent.click(toggle)
    expect(screen.queryByRole('textbox', { name: 'Search sessions' })).toBeNull()
    fireEvent.click(toggle)
    expect((screen.getByRole('textbox', { name: 'Search sessions' }) as HTMLInputElement).value).toBe('')
  })

  it('highlights the focused chat owner without switching the ambient agent', () => {
    $activeGatewayProfile.set('default')
    $selectedStoredSessionId.set('pinned')
    mount()
    const research = screen.getByRole('group', { name: /^Research ·/ })
    const defaultAgent = screen.getByRole('group', { name: /^default ·/ })
    expect(research.getAttribute('aria-current')).toBe('true')
    expect(defaultAgent.hasAttribute('aria-current')).toBe(false)

    act(() => $selectedStoredSessionId.set('regular'))
    expect(defaultAgent.getAttribute('aria-current')).toBe('true')
    expect(research.hasAttribute('aria-current')).toBe(false)
    expect($activeGatewayProfile.get()).toBe('default')
  })

  it('keeps server-pinned chats reachable and resumes the exact session', () => {
    mount()
    expect(screen.getAllByText('Pinned plan')).toHaveLength(2)

    const research = screen.getByRole('group', { name: /^Research ·/ })

    fireEvent.click(within(research).getByText('Pinned plan'))
    expect(resume).toHaveBeenCalledWith('pinned', expect.objectContaining({ profile: 'research' }))

    const agentGroup = screen.getByRole('group', { name: /^default ·/ })

    expect(within(agentGroup).getAllByRole('button', { name: 'default' })).toHaveLength(1)
    expect(within(research).getAllByRole('button', { name: 'Research' })).toHaveLength(1)
    expect(screen.queryByRole('button', { name: 'Bot Chat' })).toBeNull()
    expect(within(agentGroup).queryByText('Pinned plan')).toBeNull()
    expect(screen.queryByText('Chats', { exact: true })).toBeNull()
    expect(within(agentGroup).queryByLabelText('Agents: default')).toBeNull()
    expect(screen.getByLabelText('Agents: research')).toBeTruthy()
    fireEvent.click(screen.getByText('Regular chat'))
    expect(resume).toHaveBeenCalledWith('regular', expect.objectContaining({ profile: 'default' }))
  })
  it('hides only the chosen machine’s agent, preserves its chats in search, and restores it after remount', async () => {
    vi.spyOn(apiClient, 'hermesApi').mockResolvedValue({ results: [] })
    $sessions.set([
      makeSessionInfo({ id: 'a', title: 'Alpha notes', profile: 'research', connection_id: 'alpha', pinned: true }),
      makeSessionInfo({ id: 'b', title: 'Beta notes', profile: 'research', connection_id: 'beta' })
    ])
    $profiles.set([])
    const original = $sessions.get()
    const view = mount()
    fireEvent.keyDown(
      within(screen.getByRole('group', { name: /^research · alpha$/i })).getByRole('button', {
        name: /^Agent actions:/
      }),
      { key: 'Enter' }
    )
    fireEvent.click(await screen.findByRole('menuitem', { name: /Hide from chat list/ }))
    expect($hiddenSidebarAgents.get()).toEqual([agentSidebarKey({ connectionId: 'alpha', profile: 'research' })])
    expect(screen.queryByText('Alpha notes')).toBeNull()
    expect(screen.getByText('Beta notes')).toBeTruthy()
    expect($sessions.get()).toBe(original)
    searchFor('Alpha notes')
    fireEvent.click(screen.getAllByText('Alpha notes')[0])
    expect(resume).toHaveBeenCalledWith('a', expect.objectContaining({ connection_id: 'alpha', profile: 'research' }))
    view.unmount()
    mount()
    expect(screen.queryByText('Alpha notes')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Hidden agents/ }))
    fireEvent.click(screen.getByRole('button', { name: /^Show: research · alpha$/i }))
    expect(screen.getAllByText('Alpha notes')).toHaveLength(2)
    expect($hiddenSidebarAgents.get()).toEqual([])
    expect($sessions.get()).toBe(original)
  })

  it('retains agent expansion across navigation remounts and hide/show', () => {
    const view = mount()
    fireEvent.click(screen.getByRole('button', { name: 'Collapse: Research' }))
    expect(within(screen.getByRole('group', { name: /^Research ·/ })).queryByText('Pinned plan')).toBeNull()
    view.unmount()
    mount()
    expect(screen.getByRole('button', { name: 'Expand: Research' }).getAttribute('aria-expanded')).toBe('false')
    expect(within(screen.getByRole('group', { name: /^Research ·/ })).queryByText('Pinned plan')).toBeNull()
    act(() => setSidebarAgentHidden({ connectionId: null, profile: 'research' }, true))
    fireEvent.click(screen.getByRole('button', { name: /Hidden agents/ }))
    fireEvent.click(screen.getByRole('button', { name: /^Show: Research/ }))
    expect(screen.getByRole('button', { name: 'Expand: Research' }).getAttribute('aria-expanded')).toBe('false')
  })
  it('shows Portal pin guidance only for a settled empty list, not loading, errors, or filtering', () => {
    vi.spyOn(apiClient, 'hermesApi').mockImplementation(() => new Promise(() => {}))
    $sessions.set([makeSessionInfo({ id: 'regular', title: 'Regular chat', profile: 'default' })])
    const view = mount()
    const guidance = () => view.container.querySelector('[data-slot="pinned-empty"]')
    expect(guidance()).not.toBeNull()
    act(() => $sessionsLoading.set(true))
    expect(guidance()).toBeNull()
    act(() => {
      $sessionsLoading.set(false)
      $sessionsLoadError.set(true)
    })
    expect(guidance()).toBeNull()
    act(() => $sessionsLoadError.set(false))
    searchFor('missing')
    expect(guidance()).toBeNull()
    searchFor('')
    expect(guidance()).not.toBeNull()
    act(() => $sessions.set([makeSessionInfo({ id: 'pinned', title: 'Pinned plan', pinned: true })]))
    expect(guidance()).toBeNull()
  })
  it('opens the agent canonical chat even while its side chats are collapsed', () => {
    mount()
    fireEvent.click(screen.getByRole('button', { name: 'Collapse: Research' }))
    const agent = screen.getByRole('button', { name: 'Research' })
    fireEvent.click(agent)
    expect(openRosterBot).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'research', targetProfile: 'research', sourceScoped: false }),
      { label: 'Research' }
    )
    expect(screen.getByRole('button', { name: 'Expand: Research' }).getAttribute('aria-expanded')).toBe('false')
    expect(screen.getByTestId('route').textContent).toBe('')
    act(() => {
      $selectedStoredSessionId.set('canonical-tip')
      $openBotChat.set({
        key: 'legacy::research',
        openedRegistryId: 'canonical-root',
        openedSessionId: 'canonical-tip'
      })
    })
    expect(agent.getAttribute('aria-current')).toBe('page')
    act(() => $selectedStoredSessionId.set('side-chat'))
    expect(agent.hasAttribute('aria-current')).toBe(false)
  })
  it('opens the clicked agent settings, including agents with no chats', async () => {
    mount()
    fireEvent.keyDown(
      within(screen.getByRole('group', { name: /^Research ·/ })).getByRole('button', { name: /^Agent actions:/ }),
      { key: 'Enter' }
    )
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Open agent' }))
    await waitFor(() => expect($settingsScopeProfile.get()).toBe('research'))
    await waitFor(() => expect(screen.getByTestId('route').textContent).toBe('?agentProfile=research'))
    expect(openRosterBot).not.toHaveBeenCalled()
  })
  it('rolls back profile browsing without changing conversations, pins or active identity', () => {
    $showAllProfiles.set(false)
    $activeGatewayProfile.set('research')
    $selectedStoredSessionId.set('pinned')
    const sessions = $sessions.get()
    const pins = $pinnedSessionIds.get()
    expect($sessionListProfileScope.get()).toBe(ALL_PROFILES)
    expect($profileScope.get()).toBe('research')
    act(() => setPortalPreferred(false))
    expect($sessionListProfileScope.get()).toBe('research')
    expect($profileScope.get()).toBe('research')
    expect($sessions.get()).toBe(sessions)
    expect($pinnedSessionIds.get()).toBe(pins)
    expect($selectedStoredSessionId.get()).toBe('pinned')
    act(() => setPortalPreferred(true))
    mount()
    fireEvent.click(screen.getAllByText('Pinned plan')[0])
    expect(resume).toHaveBeenCalledWith('pinned', expect.objectContaining({ profile: 'research' }))
    expect($showAllProfiles.get()).toBe(false)
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
    const alpha = within(screen.getByRole('group', { name: 'research · Alpha' }))
    const beta = within(screen.getByRole('group', { name: 'research · Beta' }))
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

  it('keeps fetched native rows owned by local while the remote replacement is pending', async () => {
    const pending = deferred<sessionsApi.SidebarSessionsResponse>()
    const local = makeSessionInfo({ id: 'cached', title: 'Cached local', profile: 'default' })
    const mutate = vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({})
    let switched = false
    $apiRequestScope.set({ connectionId: 'local', profile: 'default' })
    vi.spyOn(apiClient, 'hermesApi').mockImplementation(request => {
      if (request.path.includes('/api/profiles/sessions/sidebar')) {
        return (
          switched
            ? pending.promise
            : Promise.resolve({
                recents: { sessions: [local] },
                cron: { sessions: [] },
                messaging: { sessions: [] }
              })
        ) as never
      }

      return Promise.resolve([]) as never
    })
    const actions = renderHook(() => useSessionListActions({ profileScope: ALL_PROFILES }))
    await act(async () => {
      await actions.result.current.refreshSessions()
    })
    mount()
    switched = true
    act(() => $apiRequestScope.set({ connectionId: 'remote-b', profile: 'default' }))
    let refresh!: Promise<void>
    act(() => {
      refresh = actions.result.current.refreshSessions()
    })
    fireEvent.click(screen.getByText('Cached local'))
    expect(resume).toHaveBeenCalledWith('cached', expect.objectContaining({ connection_id: 'local' }))
    fireEvent.keyDown(screen.getByRole('button', { name: 'Session actions: Cached local' }), { key: 'Enter' })
    fireEvent.click(await screen.findByRole('menuitem', { name: /^Pin$/ }))
    await waitFor(() =>
      expect(mutate).toHaveBeenCalledWith(
        'cached',
        'PATCH',
        { pinned: true },
        { connectionId: 'local', profile: 'default' }
      )
    )
    await act(async () => {
      pending.resolve({ recents: { sessions: [] }, cron: { sessions: [] }, messaging: { sessions: [] } })
      await refresh
    })
  })

  it.each<Record<string, number>>([{}, { 'all:telegram': 160 }])(
    'loads real messaging sources with totals %j',
    async totals => {
      $messagingTruncated.set(true)
      $messagingSessions.set([makeSessionInfo({ id: 'tg', source: 'telegram' })])
      $messagingPlatformTotals.set(totals)
      const loadMessaging = vi.fn()
      mount({ onLoadMoreMessaging: loadMessaging })
      fireEvent.click(screen.getByRole('button', { name: 'Load more' }))
      await waitFor(() => expect(loadMessaging).toHaveBeenCalledWith('telegram'))
    }
  )

  it('pages messaging through the real totals writer without sending composite keys as sources', async () => {
    const telegram = makeSessionInfo({ id: 'tg', source: 'telegram', connection_id: 'local' })
    $messagingSessions.set([telegram])
    $messagingTruncated.set(true)
    const request = vi.spyOn(apiClient, 'hermesApi').mockResolvedValue({ sessions: [telegram], total: 160 })
    const actions = renderHook(() => useSessionListActions({ profileScope: ALL_PROFILES }))
    mount({ onLoadMoreMessaging: actions.result.current.loadMoreMessagingForPlatform })

    fireEvent.click(screen.getByRole('button', { name: 'Load more' }))
    await waitFor(() => expect($messagingPlatformTotals.get()).toEqual({ 'all:telegram': 160 }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Load more' }).hasAttribute('disabled')).toBe(false))
    fireEvent.click(screen.getByRole('button', { name: 'Load more' }))
    await waitFor(() =>
      expect(request.mock.calls.filter(([call]) => call.path.includes('/api/profiles/sessions?'))).toHaveLength(2)
    )

    for (const [call] of request.mock.calls.filter(([call]) => call.path.includes('/api/profiles/sessions?'))) {
      expect(new URL(call.path, 'https://example.test').searchParams.get('source')).toBe('telegram')
    }

    await waitFor(() => expect(screen.getByRole('button', { name: 'Load more' }).hasAttribute('disabled')).toBe(false))
    act(() => $messagingPlatformTotals.set({ 'all:telegram': 1 }))
    expect(screen.queryByRole('button', { name: 'Load more' })).toBeNull()
  })

  it('starts a separate chat from an agent row without opening its canonical chat or settings', async () => {
    mount()
    fireEvent.keyDown(
      within(screen.getByRole('group', { name: /^Research ·/ })).getByRole('button', { name: /^Agent actions:/ }),
      { key: 'Enter' }
    )
    fireEvent.click(await screen.findByRole('menuitem', { name: 'New chat with this agent' }))
    expect(newSessionInProfile).toHaveBeenCalledWith('research')
    expect(screen.getByTestId('route').textContent).toBe('')
    expect(openRosterBot).not.toHaveBeenCalled()
  })

  it('offers Unpin under the agent as well as in Pinned, keeping the owning profile', async () => {
    const update = vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({})

    try {
      mount()

      const research = screen.getByRole('group', { name: /^Research ·/ })

      fireEvent.keyDown(within(research).getByRole('button', { name: /^Session actions:/ }), { key: 'Enter' })
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

      const research = screen.getByRole('group', { name: /^Research ·/ })

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
    const alphaToggle = within(alpha).getByRole('button', { name: 'Collapse: Alpha' })
    const betaToggle = within(beta).getByRole('button', { name: 'Collapse: Beta' })
    fireEvent.click(alphaToggle)
    fireEvent.click(betaToggle)
    fireEvent.click(screen.getByText('Pinned plan'))
    expect(alphaToggle.getAttribute('aria-expanded')).toBe('true')
    expect(betaToggle.getAttribute('aria-expanded')).toBe('false')
    expect(screen.getAllByText('Pinned plan')).toHaveLength(2)
  })

  it('preserves manual agent expansion when selecting chats and only reveals a pinned chat owner', () => {
    mount()
    const research = screen.getByRole('button', { name: 'Collapse: Research', expanded: true })
    const defaultAgent = screen.getByRole('button', { name: 'Collapse: default', expanded: true })
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

  it('opens a real cross-gateway search-only result with its immutable tab owner', async () => {
    $apiRequestScope.set({ connectionId: 'local', profile: 'default' })
    $profiles.set([])
    $sessionTiles.set([])
    $fleetRoster.set({
      sources: [{ connectionId: 'remote-b', kind: 'remote', label: 'Beta', reachable: true }],
      agents: [
        {
          connectionId: 'remote-b',
          connectionKind: 'remote',
          connectionLabel: 'Beta',
          profile: 'research',
          handle: 'research'
        }
      ]
    })
    vi.spyOn(apiClient, 'hermesApi').mockResolvedValue({
      results: [
        { session_id: 'search-only-remote', title: 'Remote search result', snippet: 'remote', session_started: 1 }
      ]
    })
    mount()
    searchFor('remote')
    fireEvent.click(await screen.findByText('Remote search result'), { ctrlKey: true })
    expect($sessions.get().some(row => row.id === 'search-only-remote')).toBe(false)
    expect($sessionTiles.get()).toMatchObject([
      {
        storedSessionId: 'search-only-remote',
        ownerRoute: { connectionId: 'remote-b', profile: 'research' },
        workspaceMode: 'sessions'
      }
    ])
  })

  it('shows loading rather than no matches while a search is pending', () => {
    vi.spyOn(apiClient, 'hermesApi').mockImplementation(() => new Promise(() => {}))

    try {
      mount()
      searchFor('not loaded yet')
      expect(screen.getByText('Loading…', { selector: '[role=status]' }).getAttribute('role')).toBe('status')
      expect(screen.queryByText(/No sessions match/)).toBeNull()
    } finally {
      vi.restoreAllMocks()
    }
  })

  it('withholds bare-ID actions for search-only owner collisions and restores them when search clears', async () => {
    vi.spyOn(apiClient, 'hermesApi')
      .mockResolvedValueOnce({ results: [] })
      .mockResolvedValue({
        results: [{ session_id: 'regular', title: 'Other owner chat', snippet: 'other', session_started: 1 }]
      })
    mount()
    searchFor('other')
    await screen.findByText('Other owner chat')
    expect(screen.queryByRole('button', { name: 'Session actions: Other owner chat' })).toBeNull()
    fireEvent.click(screen.getByText('Other owner chat'))
    expect(resume).toHaveBeenCalledWith('regular', expect.objectContaining({ profile: 'research' }))
    searchFor('')
    expect(screen.getByRole('button', { name: 'Session actions: Regular chat' })).toBeTruthy()
  })

  it('keeps a newly pinned search result visible after clearing search, before the list refreshes', async () => {
    vi.spyOn(apiClient, 'hermesApi')
      .mockResolvedValueOnce({ results: [] })
      .mockResolvedValue({
        results: [{ session_id: 'older', title: 'Older plan', snippet: 'older', session_started: 1 }]
      })
    let resolve!: (value: unknown) => void
    vi.spyOn(sessionsApi, 'mutateSessionHttp').mockImplementation(
      () =>
        new Promise(done => {
          resolve = done
        })
    )

    try {
      mount()
      searchFor('older')
      await screen.findByText('Older plan')

      const research = screen.getByRole('group', { name: /^Research ·/ })

      fireEvent.keyDown(within(research).getByRole('button', { name: /^Session actions:/ }), { key: 'Enter' })
      fireEvent.click(await screen.findByRole('menuitem', { name: /^Pin$/ }))
      expect(screen.getAllByText('Older plan')).toHaveLength(2)
      searchFor('')
      expect(screen.getAllByText('Older plan')).toHaveLength(2)
      await act(async () => {
        resolve({ ok: true })
      })
      act(() =>
        $sessions.set([
          ...$sessions.get(),
          makeSessionInfo({ id: 'older', title: 'Older plan', profile: 'research', pinned: true })
        ])
      )
      expect(screen.getAllByText('Older plan')).toHaveLength(2)
    } finally {
      vi.restoreAllMocks()
    }
  })

  it('renders every pin, caps an agent at its recent chats behind Show all, and searches past the cap', () => {
    $sessions.set(
      Array.from({ length: 9 }, (_, index) =>
        makeSessionInfo({
          id: `saved-${index}`,
          title: `Saved ${index}`,
          profile: 'research',
          pinned: true,
          last_active: 100 - index
        })
      )
    )
    vi.spyOn(apiClient, 'hermesApi').mockResolvedValue({ results: [] })
    mount()
    const pins = within(screen.getByRole('region', { name: 'Pinned' }))

    const agent = within(screen.getByRole('group', { name: /^Research ·/ }))

    expect(pins.getAllByText(/^Saved /)).toHaveLength(9)
    expect(agent.getAllByText(/^Saved /)).toHaveLength(3)
    expect(screen.queryByRole('button', { name: /^Show .* more/ })).toBeNull()
    searchFor('Saved 8')
    expect(pins.getAllByText(/^Saved /)).toHaveLength(1)
    expect(agent.getByText('Saved 8')).toBeTruthy()
    fireEvent.click(pins.getByText('Saved 8'))
    expect(resume).toHaveBeenCalledWith('saved-8', expect.objectContaining({ profile: 'research' }))
    searchFor('')
    expect(agent.getAllByText(/^Saved /)).toHaveLength(3)
    fireEvent.click(agent.getByRole('button', { name: 'Show all (9)' }))
    expect(agent.getAllByText(/^Saved /)).toHaveLength(9)
    fireEvent.click(agent.getByRole('button', { name: 'Show less' }))
    expect(agent.getAllByText(/^Saved /)).toHaveLength(3)
  })

  it('marks pinned agent chats and sorts them first, restoring recency when unpinned', async () => {
    vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({})
    $sessions.set([
      makeSessionInfo({ id: 'newest', title: 'Newest', profile: 'research', last_active: 400 }),
      makeSessionInfo({ id: 'pin-newer', title: 'Pinned newer', profile: 'research', last_active: 300, pinned: true }),
      makeSessionInfo({ id: 'pin-older', title: 'Pinned older', profile: 'research', last_active: 200, pinned: true }),
      makeSessionInfo({ id: 'oldest', title: 'Oldest', profile: 'research', last_active: 100 })
    ])
    mount()

    const agent = within(screen.getByRole('group', { name: /^Research ·/ }))

    const titles = () => agent.getAllByText(/^(Newest|Pinned newer|Pinned older|Oldest)$/).map(row => row.textContent)
    // Pinned chats lead, so the card's cap keeps them in view.
    expect(titles()).toEqual(['Pinned newer', 'Pinned older', 'Newest'])
    fireEvent.click(agent.getByRole('button', { name: 'Show all (4)' }))
    expect(titles()).toEqual(['Pinned newer', 'Pinned older', 'Newest', 'Oldest'])
    expect(agent.getAllByRole('img', { name: 'Pinned' })).toHaveLength(2)
    const pins = within(screen.getByRole('region', { name: 'Pinned' }))
    expect(pins.getByText('Pinned older')).toBeTruthy()
    expect(pins.queryByRole('img', { name: 'Pinned' })).toBeNull()
    fireEvent.keyDown(agent.getByRole('button', { name: 'Session actions: Pinned older' }), { key: 'Enter' })
    fireEvent.click(await screen.findByRole('menuitem', { name: /Unpin/ }))
    expect(titles()).toEqual(['Pinned newer', 'Newest', 'Pinned older', 'Oldest'])
    expect(agent.getAllByRole('img', { name: 'Pinned' })).toHaveLength(1)
    expect(pins.queryByText('Pinned older')).toBeNull()
  })

  it.each([
    ['below', 350, 324],
    ['above', 20, 80],
    ['visible', 180, 200]
  ])(
    'reveals a pinned chat %s the Agents viewport without moving other scroll regions',
    (_position, rowTop, expected) => {
      mount()
      const agents = screen.getByRole('region', { name: 'Agents' })
      const pins = screen.getByRole('region', { name: 'Pinned' })
      const scroller = agents.querySelector<HTMLElement>('.scroll-fade-y')!
      const pinScroller = pins.querySelector<HTMLElement>('.scroll-fade-y')!
      const pin = within(pins).getByText('Pinned plan').closest('button')!
      Object.defineProperty(scroller, 'clientHeight', { configurable: true, value: 200 })
      scroller.scrollTop = 200
      pinScroller.scrollTop = 33
      const nativeRect = HTMLElement.prototype.getBoundingClientRect
      vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
        if (this === scroller) {
          return { top: 100, bottom: 300 } as DOMRect
        }

        if (agents.contains(this) && this.dataset.sessionKey && this.textContent?.includes('Pinned plan')) {
          return { top: rowTop, bottom: rowTop + 34 } as DOMRect
        }

        return nativeRect.call(this)
      })

      fireEvent.click(screen.getByRole('button', { name: 'Collapse: Research' }))
      pin.focus()
      fireEvent.click(pin)
      expect(screen.getByRole('button', { name: 'Collapse: Research' })).toBeTruthy()
      expect(scroller.scrollTop).toBe(expected)
      expect(pinScroller.scrollTop).toBe(33)
      expect(window.document.activeElement).toBe(pin)

      // Background list refreshes must not repeat a completed reveal.
      scroller.scrollTop = 200
      act(() => $sessions.set([...$sessions.get()]))
      expect(scroller.scrollTop).toBe(200)
      fireEvent.click(within(agents).getByText('Regular chat'))
      expect(scroller.scrollTop).toBe(200)

      // Explicitly clicking the same pin again is a new reveal request.
      fireEvent.click(pin)
      expect(scroller.scrollTop).toBe(expected)
    }
  )

  it('marks only chats with live status, so settled rows carry no dot', () => {
    $sessions.set([
      makeSessionInfo({ id: 'pinned', title: 'Pinned plan', profile: 'research', pinned: true, unread: true }),
      makeSessionInfo({ id: 'regular', title: 'Regular chat', profile: 'default' })
    ])
    mount()

    const row = (group: RegExp, title: string) =>
      within(screen.getByRole('group', { name: group }))
        .getByText(title)
        .closest<HTMLElement>('[data-session-key]')!

    expect(within(row(/^Research ·/, 'Pinned plan')).getByRole('status', { name: 'Finished — unread' })).toBeTruthy()
    expect(within(row(/^default ·/, 'Regular chat')).queryByRole('status')).toBeNull()
  })

  it('walks sidebar rows with arrow, Home and End keys', () => {
    mount()
    const pinned = screen.getByRole('button', { name: 'Pinned', expanded: true })
    pinned.focus()

    fireEvent.keyDown(pinned, { key: 'ArrowDown' })
    expect(window.document.activeElement?.textContent).toContain('Pinned plan')
    fireEvent.keyDown(window.document.activeElement!, { key: 'ArrowUp' })
    expect(window.document.activeElement).toBe(pinned)
    fireEvent.keyDown(pinned, { key: 'End' })
    expect(window.document.activeElement).not.toBe(pinned)
    fireEvent.keyDown(window.document.activeElement!, { key: 'Home' })
    expect(window.document.activeElement).toBe(pinned)
  })

  it('collapses the Agents section on its own preference, leaving classic sections untouched', () => {
    const view = mount()
    const toggle = screen.getByRole('button', { name: 'Agents', expanded: true })

    fireEvent.click(toggle)
    expect(screen.queryByRole('group', { name: /^Research ·/ })).toBeNull()
    expect($portalAgentsSectionOpen.get()).toBe(false)
    expect($sidebarRecentsOpen.get()).toBe(true)

    view.unmount()
    mount()
    fireEvent.click(screen.getByRole('button', { name: 'Agents', expanded: false }))
    expect(screen.getByRole('group', { name: /^Research ·/ })).toBeTruthy()
  })

  it('collapses the Pinned section on its own preference, leaving classic Pinned untouched', () => {
    const view = mount()

    const pinnedList = () => within(screen.getByRole('region', { name: 'Pinned' })).queryByText('Pinned plan')

    expect(pinnedList()).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Pinned', expanded: true }))
    expect(pinnedList()).toBeNull()
    expect($portalPinnedSectionOpen.get()).toBe(false)
    expect($sidebarPinsOpen.get()).toBe(true)

    view.unmount()
    mount()
    fireEvent.click(screen.getByRole('button', { name: 'Pinned', expanded: false }))
    expect(pinnedList()).toBeTruthy()
  })
})
