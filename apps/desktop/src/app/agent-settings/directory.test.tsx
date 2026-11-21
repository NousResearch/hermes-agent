import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, type RenderOptions, screen, waitFor, within } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { queryClient } from '@/lib/query-client'
import type * as FleetRosterStore from '@/store/fleet-roster'
import { refreshFleetRoster } from '@/store/fleet-roster'
import type * as ProfileStore from '@/store/profile'
import { $profiles, refreshProfiles } from '@/store/profile'

import type { SimplifiedAgent } from '../chat/sidebar/use-simplified-agents'

import { AgentDirectory, AgentNavigation } from './directory'

let narrow = false

vi.mock('@/hooks/use-media-query', () => ({ useMediaQuery: () => narrow }))
vi.mock('../chat/sidebar/use-fleet-roster', () => ({ useFleetRoster: () => {} }))
const roster: { agents: SimplifiedAgent[] } = { agents: [] }

const twoAgents: SimplifiedAgent[] = [
  { id: 'a', label: 'Research', profile: 'research', connectionId: 'alpha', gateway: 'Laptop' },
  { id: 'b', label: 'Research', profile: 'research', connectionId: 'beta', gateway: 'Studio' }
]

vi.mock('../chat/sidebar/use-simplified-agents', () => ({
  useSimplifiedAgents: () => ({ owner: 'alpha', agents: roster.agents })
}))
vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<typeof ProfileStore>()),
  refreshProfiles: vi.fn(async () => [])
}))
vi.mock('@/store/fleet-roster', async importOriginal => ({
  ...(await importOriginal<typeof FleetRosterStore>()),
  refreshFleetRoster: vi.fn(async () => undefined)
}))
beforeEach(() => {
  roster.agents = twoAgents
})
afterEach(() => {
  cleanup()
  queryClient.clear()
  vi.clearAllMocks()
  $profiles.set([])
})

const withQueries: RenderOptions['wrapper'] = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
)

function Route() {
  const location = useLocation()

  return (
    <output aria-label="route">
      {location.pathname}
      {location.search}
    </output>
  )
}

it('lands a normal window on an agent page instead of a directory', () => {
  narrow = false
  render(
    <MemoryRouter>
      <AgentDirectory onClose={vi.fn()} />
      <Route />
    </MemoryRouter>,
    { wrapper: withQueries }
  )
  expect(screen.getByLabelText('route').textContent).toBe('/settings?agentProfile=research&connection=alpha')
})

it('lists agents in a narrow window and opens the chosen gateway/profile identity', () => {
  narrow = true
  render(
    <MemoryRouter>
      <AgentDirectory onClose={vi.fn()} />
      <Route />
    </MemoryRouter>,
    { wrapper: withQueries }
  )
  // A two-agent roster is scanned, not searched.
  expect(screen.queryByRole('textbox')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Research · Studio' }))
  expect(screen.getByLabelText('route').textContent).toBe('/settings?agentProfile=research&connection=beta')
})

it('opens agent creation from the narrow list without navigating', () => {
  narrow = true
  render(
    <MemoryRouter>
      <AgentDirectory onClose={vi.fn()} />
      <Route />
    </MemoryRouter>,
    { wrapper: withQueries }
  )
  fireEvent.click(screen.getByRole('button', { name: en.profiles.newProfile }))
  expect(screen.getByRole('dialog', { name: en.profiles.newProfile })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: en.common.cancel }))
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(screen.getByLabelText('route').textContent).toBe('/')
})

it.each([true, false])('keeps default identity attached to its machine (is_default=%s)', isDefault => {
  $profiles.set([
    {
      name: 'research',
      path: '/alpha/research',
      has_env: false,
      is_default: isDefault,
      model: null,
      provider: null,
      skill_count: 0
    }
  ])
  render(
    <MemoryRouter>
      <AgentNavigation connectionId="alpha" profile="research" />
    </MemoryRouter>,
    { wrapper: withQueries }
  )
  const local = screen.getByRole('button', { name: 'Research · Laptop' })
  const remote = screen.getByRole('button', { name: 'Research · Studio' })

  expect(Boolean(within(local).queryByText(en.profiles.defaultBadge))).toBe(isDefault)
  expect(within(remote).queryByText(en.profiles.defaultBadge)).toBeNull()
})

it.each(['directory', 'navigation'])(
  'ends an empty roster that failed to load in a Retry, not a loader (%s)',
  async at => {
    narrow = false
    roster.agents = []
    vi.mocked(refreshProfiles).mockRejectedValueOnce(new Error('gateway unreachable'))
    render(
      <MemoryRouter>
        {at === 'directory' ? (
          <AgentDirectory onClose={vi.fn()} />
        ) : (
          <AgentNavigation connectionId="alpha" profile="research" />
        )}
      </MemoryRouter>,
      { wrapper: withQueries }
    )

    expect(await screen.findByText('gateway unreachable')).toBeTruthy()
    expect(refreshFleetRoster).toHaveBeenCalledWith({ force: true })
    roster.agents = twoAgents
    fireEvent.click(screen.getByRole('button', { name: en.common.retry }))
    await waitFor(() => expect(screen.queryByText('gateway unreachable')).toBeNull())
    expect(refreshProfiles).toHaveBeenCalledTimes(2)
  }
)
