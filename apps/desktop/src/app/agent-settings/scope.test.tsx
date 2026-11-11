import type * as ReactQuery from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import type { MessagingPlatformsResponse } from '@/hermes'
import { openRosterBot } from '@/plugins/hermes-bots/roster-actions'
import { $fleetRoster } from '@/store/fleet-roster'
import { $activeGatewayProfile } from '@/store/profile'
import { $settingsScopeOverride } from '@/store/settings-scope'

import { openSession } from '../open-session'

import { AgentSettingsView } from './index'

const { queryData } = vi.hoisted(() => ({
  queryData: {
    sessions: [] as { id: string; title: string; hidden?: boolean }[],
    messaging: undefined as MessagingPlatformsResponse | undefined
  }
}))

vi.mock('@tanstack/react-query', async importOriginal => ({
  ...(await importOriginal<typeof ReactQuery>()),
  useQuery: ({ queryKey }: { queryKey: readonly unknown[] }) =>
    queryKey.includes('recent')
      ? { data: { sessions: queryData.sessions } }
      : queryKey.includes('messaging')
        ? { data: queryData.messaging }
        : {}
}))
vi.mock('../hooks/use-config-record', () => ({ useHermesConfigRecord: () => ({}) }))
vi.mock('../capabilities/skills/skills-data', () => ({ useSkillsQuery: () => ({}) }))
vi.mock('../capabilities/toolsets/toolsets-data', () => ({ useToolsetsQuery: () => ({}) }))
vi.mock('@/plugins/hermes-bots/roster-actions', () => ({ openRosterBot: vi.fn(async () => true) }))
vi.mock('../open-session', () => ({ openSession: vi.fn() }))

afterEach(() => {
  cleanup()
  $apiRequestScope.set({ profile: null, connectionId: null })
  $activeGatewayProfile.set('default')
  $fleetRoster.set(null)
  $settingsScopeOverride.set(null)
  queryData.sessions = []
  queryData.messaging = undefined
  vi.clearAllMocks()
})

it('keeps canonical chat and recent sessions on their owner and removes them when the owner changes', async () => {
  $activeGatewayProfile.set('research')
  $apiRequestScope.set({ profile: 'research', connectionId: 'alpha' })
  $fleetRoster.set({
    sources: [],
    agents: [
      {
        profile: 'research',
        targetProfile: 'other',
        connectionId: 'beta',
        connectionKind: 'remote',
        connectionLabel: 'Beta',
        handle: 'research-beta'
      },
      {
        profile: 'research',
        targetProfile: 'default',
        connectionId: 'alpha',
        connectionKind: 'remote',
        connectionLabel: 'Alpha',
        handle: 'research-alpha'
      }
    ]
  })
  queryData.sessions = [
    { id: 'side-chat', title: 'Research notes' },
    { id: 'canonical', title: 'Bot Chat', hidden: true }
  ]
  const onClose = vi.fn()
  render(
    <MemoryRouter initialEntries={['/settings?agentProfile=research&connection=alpha']}>
      <AgentSettingsView onClose={onClose} owner={{ connectionId: 'alpha', profile: 'research' }} />
    </MemoryRouter>
  )
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Open chat' })))
  expect(openRosterBot).toHaveBeenCalledWith(
    expect.objectContaining({
      name: 'research',
      targetProfile: 'default',
      connectionId: 'alpha',
      sourceScoped: true
    })
  )
  expect(onClose).toHaveBeenCalledOnce()
  expect(screen.queryByRole('button', { name: 'Bot Chat' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Research notes' }))
  expect(openSession).toHaveBeenCalledWith('side-chat', expect.any(Function), 'in-place', {
    workspaceMode: 'sessions',
    ownerRoute: { connectionId: 'alpha', profile: 'research' }
  })
  act(() => $apiRequestScope.setKey('connectionId', 'beta'))
  expect(screen.queryByRole('button', { name: 'Research notes' })).toBeNull()
  expect(screen.getByRole('button', { name: 'Open agent' })).toBeTruthy()
})

it.each(['fatal', 'startup_failed', 'retrying', 'gateway_stopped', 'stopped', 'disconnected', 'error', 'failed'])(
  'warns about an enabled configured channel in %s state in both overview surfaces',
  state => {
    $apiRequestScope.set({ profile: 'default', connectionId: 'local' })
    queryData.messaging = {
      platforms: [{ id: 'telegram', name: 'Telegram', configured: true, enabled: true, state }]
    } as MessagingPlatformsResponse
    render(
      <MemoryRouter initialEntries={['/settings?agentProfile=default&connection=local']}>
        <AgentSettingsView onClose={vi.fn()} owner={{ connectionId: 'local', profile: 'default' }} />
      </MemoryRouter>
    )
    expect(screen.getByRole('alert').textContent).toContain('Telegram')
    expect(screen.getByRole('button', { name: /Telegram/ }).textContent).toContain('Disconnected')
    expect(screen.getByRole('button', { name: /Telegram/ }).textContent).not.toContain('Configured')
  }
)

it.each([
  { state: 'connected', enabled: true, expected: 'Connected' },
  { state: 'fatal', enabled: false, expected: 'Configured' }
])('does not warn about $state channels when enabled=$enabled', ({ state, enabled, expected }) => {
  $apiRequestScope.set({ profile: 'default', connectionId: 'local' })
  queryData.messaging = {
    platforms: [{ id: 'telegram', name: 'Telegram', configured: true, enabled, state }]
  } as MessagingPlatformsResponse
  render(
    <MemoryRouter initialEntries={['/settings?agentProfile=default&connection=local']}>
      <AgentSettingsView onClose={vi.fn()} owner={{ connectionId: 'local', profile: 'default' }} />
    </MemoryRouter>
  )
  expect(screen.queryByRole('alert')).toBeNull()
  expect(screen.getByRole('button', { name: /Telegram/ }).textContent).toContain(expected)
})
