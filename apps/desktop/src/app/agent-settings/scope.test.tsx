import type * as ReactQuery from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { openRosterBot } from '@/plugins/hermes-bots/roster-actions'
import { $fleetRoster } from '@/store/fleet-roster'
import { $activeGatewayProfile } from '@/store/profile'
import { $settingsScopeOverride, $settingsScopeProfile } from '@/store/settings-scope'

import { openSession } from '../open-session'

import { AgentSettingsView } from './index'

const { queryData } = vi.hoisted(() => ({ queryData: { sessions: [] as { id: string; title: string; hidden?: boolean }[] } }))

vi.mock('@tanstack/react-query', async importOriginal => ({
  ...await importOriginal<typeof ReactQuery>(),
  useQuery: () => ({}),
  useInfiniteQuery: () => ({
    data: { pages: [{ sessions: queryData.sessions, total: queryData.sessions.length, limit: 25, offset: 0 }] }
  })
}))
vi.mock('../hooks/use-config-record', () => ({ useHermesConfigRecord: () => ({}) }))
vi.mock('../capabilities/skills/skills-data', () => ({ useSkillsQuery: () => ({}) }))
vi.mock('../capabilities/toolsets/toolsets-data', () => ({ useToolsetsQuery: () => ({}) }))
vi.mock('../capabilities', () => ({
  CapabilitiesView: ({ fixedProfile, fixedConnection }: { fixedProfile: string; fixedConnection: string }) => (
    <div>Capabilities {fixedConnection}/{fixedProfile} {useLocation().search}</div>
  )
}))
vi.mock('../messaging', () => ({ MessagingView: () => null }))
vi.mock('../settings', () => ({ SettingsView: () => <div>Editing {$settingsScopeProfile.get()}</div> }))
vi.mock('@/plugins/hermes-bots/roster-actions', () => ({ openRosterBot: vi.fn(async () => true) }))
vi.mock('../open-session', () => ({ openSession: vi.fn() }))

afterEach(() => {
  cleanup()
  $apiRequestScope.set({ profile: null, connectionId: null })
  $activeGatewayProfile.set('default')
  $fleetRoster.set(null)
  $settingsScopeOverride.set(null)
  queryData.sessions = []
  vi.clearAllMocks()
})

it('sets the profile before mounting a deep-linked editor and unmounts it when the gateway changes', () => {
  $activeGatewayProfile.set('research')
  $apiRequestScope.set({ profile: 'research', connectionId: 'alpha' })
  $settingsScopeOverride.set('other')
  render(
    <MemoryRouter initialEntries={['/settings?agent=research&connection=alpha&panel=settings']}>
      <AgentSettingsView onClose={() => {}} owner={{ connectionId: 'alpha', profile: 'research' }} />
    </MemoryRouter>
  )
  expect(screen.getByText('Editing research')).toBeTruthy()
  expect(screen.queryByText('Editing other')).toBeNull()
  act(() => $apiRequestScope.setKey('connectionId', 'beta'))
  expect(screen.queryByText('Editing research')).toBeNull()
  expect(screen.getByRole('button', { name: 'Open agent' })).toBeTruthy()
})

it('opens Plugins for the selected agent and removes its editor when the owner changes', () => {
  $activeGatewayProfile.set('research')
  $apiRequestScope.set({ profile: 'research', connectionId: 'alpha' })
  render(
    <MemoryRouter initialEntries={['/settings?agent=research&connection=alpha']}>
      <AgentSettingsView onClose={() => {}} owner={{ connectionId: 'alpha', profile: 'research' }} />
    </MemoryRouter>
  )
  fireEvent.click(screen.getByRole('button', { name: 'Plugins' }))
  expect(screen.getByText('Capabilities alpha/research ?agent=research&connection=alpha&panel=capabilities&tab=plugins')).toBeTruthy()
  act(() => $activeGatewayProfile.set('default'))
  expect(screen.queryByText(/Capabilities alpha/)).toBeNull()
  expect(screen.getByRole('button', { name: 'Open agent' })).toBeTruthy()
})

it('keeps the canonical chat and history sessions on the selected owner', async () => {
  $activeGatewayProfile.set('research')
  $apiRequestScope.set({ profile: 'research', connectionId: 'alpha' })
  $fleetRoster.set({
    sources: [],
    agents: [
      { profile: 'research', targetProfile: 'other', connectionId: 'beta', connectionKind: 'remote', connectionLabel: 'Beta', handle: 'research-beta' },
      { profile: 'research', targetProfile: 'default', connectionId: 'alpha', connectionKind: 'remote', connectionLabel: 'Alpha', handle: 'research-alpha' }
    ]
  })
  queryData.sessions = [
    { id: 'side-chat', title: 'Research notes' },
    { id: 'canonical', title: 'Bot Chat', hidden: true }
  ]
  const onClose = vi.fn()
  render(
    <MemoryRouter initialEntries={['/settings?agent=research&connection=alpha']}>
      <AgentSettingsView onClose={onClose} owner={{ connectionId: 'alpha', profile: 'research' }} />
    </MemoryRouter>
  )
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Open chat' })))
  expect(openRosterBot).toHaveBeenCalledWith(expect.objectContaining({
    name: 'research', targetProfile: 'default', connectionId: 'alpha', sourceScoped: true
  }))
  expect(onClose).toHaveBeenCalledOnce()
  fireEvent.click(screen.getByRole('button', { name: /^History$/ }))
  expect(screen.queryByRole('button', { name: 'Bot Chat' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Research notes' }))
  expect(openSession).toHaveBeenCalledWith('side-chat', expect.any(Function), 'in-place', {
    workspaceMode: 'sessions', ownerRoute: { connectionId: 'alpha', profile: 'research' }
  })
  fireEvent.click(screen.getByRole('button', { name: /^Settings$/ }))
  expect(screen.getByText('Editing research')).toBeTruthy()
})
