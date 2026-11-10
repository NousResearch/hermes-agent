import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'

import { AgentDirectory } from './directory'

vi.mock('../chat/sidebar/use-fleet-roster', () => ({ useFleetRoster: () => {} }))
vi.mock('../chat/sidebar/use-simplified-agents', () => ({
  useSimplifiedAgents: () => ({ agents: [
    { id: 'a', label: 'Research', profile: 'research', connectionId: 'alpha', gateway: 'Laptop' },
    { id: 'b', label: 'Research', profile: 'research', connectionId: 'beta', gateway: 'Studio' }
  ] })
}))
afterEach(cleanup)

function Route() {
  const location = useLocation()

  return <output aria-label="route">{location.pathname}{location.search}</output>
}

it('opens the selected gateway/profile identity even when agent names match', () => {
  render(<MemoryRouter><AgentDirectory onClose={vi.fn()} /><Route /></MemoryRouter>)
  fireEvent.click(screen.getByRole('button', { name: `${en.simplifiedAgents.settings}: Research · Studio` }))
  expect(screen.getByLabelText('route').textContent).toBe('/settings?agentProfile=research&connection=beta')
})

it('filters by gateway and retains the profile management destination', () => {
  render(<MemoryRouter><AgentDirectory onClose={vi.fn()} /><Route /></MemoryRouter>)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Laptop' } })
  expect(screen.queryByRole('button', { name: `${en.simplifiedAgents.settings}: Research · Studio` })).toBeNull()
  expect(screen.getByRole('button', { name: `${en.simplifiedAgents.settings}: Research · Laptop` })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: en.profiles.manageProfiles }))
  expect(screen.getByLabelText('route').textContent).toBe('/profiles?manage=true')
})

it('explains an empty search without asking users to select a missing profile', () => {
  render(<MemoryRouter><AgentDirectory onClose={vi.fn()} /></MemoryRouter>)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'missing gateway' } })
  expect(screen.queryByRole('button', { name: `${en.simplifiedAgents.settings}: Research · Laptop` })).toBeNull()
  expect(screen.queryByRole('button', { name: `${en.simplifiedAgents.settings}: Research · Studio` })).toBeNull()
})
