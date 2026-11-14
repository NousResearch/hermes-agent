import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $profiles } from '@/store/profile'

import { AgentDirectory, AgentNavigation } from './directory'

let narrow = false

vi.mock('@/hooks/use-media-query', () => ({ useMediaQuery: () => narrow }))
vi.mock('../chat/sidebar/use-fleet-roster', () => ({ useFleetRoster: () => {} }))
vi.mock('../chat/sidebar/use-simplified-agents', () => ({
  useSimplifiedAgents: () => ({
    owner: 'alpha',
    agents: [
      { id: 'a', label: 'Research', profile: 'research', connectionId: 'alpha', gateway: 'Laptop' },
      { id: 'b', label: 'Research', profile: 'research', connectionId: 'beta', gateway: 'Studio' }
    ]
  })
}))
afterEach(() => {
  cleanup()
  $profiles.set([])
})

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
    </MemoryRouter>
  )
  expect(screen.getByLabelText('route').textContent).toBe('/settings?agentProfile=research&connection=alpha')
})

it('lists agents in a narrow window and opens the chosen gateway/profile identity', () => {
  narrow = true
  render(
    <MemoryRouter>
      <AgentDirectory onClose={vi.fn()} />
      <Route />
    </MemoryRouter>
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
    </MemoryRouter>
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
    </MemoryRouter>
  )
  const local = screen.getByRole('button', { name: 'Research · Laptop' })
  const remote = screen.getByRole('button', { name: 'Research · Studio' })

  expect(Boolean(within(local).queryByText(en.profiles.defaultBadge))).toBe(isDefault)
  expect(within(remote).queryByText(en.profiles.defaultBadge)).toBeNull()
})
