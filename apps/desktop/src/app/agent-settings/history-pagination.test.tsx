import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type * as ReactQuery from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { $activeGatewayProfile } from '@/store/profile'
import { $settingsScopeOverride } from '@/store/settings-scope'
import { makeSessionInfo } from '@/test/session-info'

import type * as AgentData from './data'

import { AgentSettingsView } from './index'

const { requests } = vi.hoisted(() => ({ requests: [] as { limit: number; offset: number }[] }))

vi.mock('@tanstack/react-query', async importOriginal => {
  const original = await importOriginal<typeof ReactQuery>()

  return {
    ...original,
    useQuery: (options: ReactQuery.UseQueryOptions) => options.queryKey.includes('recent') ? original.useQuery(options) : {}
  }
})
vi.mock('../hooks/use-config-record', () => ({ useHermesConfigRecord: () => ({}) }))
vi.mock('../capabilities/skills/skills-data', () => ({ useSkillsQuery: () => ({}) }))
vi.mock('../capabilities/toolsets/toolsets-data', () => ({ useToolsetsQuery: () => ({}) }))
vi.mock('../capabilities', () => ({ CapabilitiesView: () => null }))
vi.mock('../messaging', () => ({ MessagingView: () => null }))
vi.mock('../settings', () => ({ SettingsView: () => null }))
vi.mock('./directory', () => ({ AgentNavigation: () => null }))
vi.mock('./data', async importOriginal => ({
  ...await importOriginal<typeof AgentData>(),
  getAgentRecentSessions: async (_owner: unknown, limit = 5, offset = 0) => {
    requests.push({ limit, offset })

    if (limit > 500) {throw new Error('HTTP 422: limit must be at most 500')}

    const sessions = Array.from({ length: Math.min(limit, 501 - offset) }, (_, index) => makeSessionInfo({
      id: `history-${offset + index}`, title: `History entry ${offset + index}`, profile: 'research'
    }))

    // The real endpoint backfills later pinned sessions outside each requested window.
    if (offset + limit <= 499) {
      sessions.push(makeSessionInfo({ id: 'history-499', title: 'History entry 499', profile: 'research', pinned: true }))
    }

    return { limit, offset, total: 501, sessions }
  }
}))

afterEach(() => {
  cleanup()
  $apiRequestScope.set({ profile: null, connectionId: null })
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
  requests.length = 0
})

it('keeps earlier history reachable while loading beyond the server’s 500-session request bound', async () => {
  $activeGatewayProfile.set('research')
  $apiRequestScope.set({ profile: 'research', connectionId: 'alpha' })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/settings?agent=research&connection=alpha&panel=history']}>
        <AgentSettingsView onClose={() => {}} owner={{ connectionId: 'alpha', profile: 'research' }} />
      </MemoryRouter>
    </QueryClientProvider>
  )
  await screen.findByText('History entry 24')

  for (let offset = 25; offset <= 500; offset += 25) {
    fireEvent.click(screen.getByText('Load more'))
    await waitFor(() => expect(screen.getByText(`History entry ${offset === 500 ? 500 : offset + 23}`)).toBeTruthy())
  }

  expect(screen.getByText('History entry 0')).toBeTruthy()
  expect(screen.getAllByText('History entry 499')).toHaveLength(1)
  expect(screen.queryByText('Load more')).toBeNull()
  expect(requests.every(request => request.limit <= 500)).toBe(true)
  client.clear()
}, 30_000)
