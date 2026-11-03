import type * as ReactQuery from '@tanstack/react-query'
import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { $activeGatewayProfile } from '@/store/profile'
import { $settingsScopeOverride, $settingsScopeProfile } from '@/store/settings-scope'

import { AgentSettingsView } from './index'

vi.mock('@tanstack/react-query', async importOriginal => ({
  ...await importOriginal<typeof ReactQuery>(),
  useQuery: () => ({})
}))
vi.mock('../hooks/use-config-record', () => ({ useHermesConfigRecord: () => ({}) }))
vi.mock('../capabilities/skills/skills-data', () => ({ useSkillsQuery: () => ({}) }))
vi.mock('../capabilities/toolsets/toolsets-data', () => ({ useToolsetsQuery: () => ({}) }))
vi.mock('../capabilities', () => ({ CapabilitiesView: () => null }))
vi.mock('../messaging', () => ({ MessagingView: () => null }))
vi.mock('../settings', () => ({ SettingsView: () => <div>Editing {$settingsScopeProfile.get()}</div> }))

afterEach(() => {
  cleanup()
  $apiRequestScope.set({ profile: null, connectionId: null })
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
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
