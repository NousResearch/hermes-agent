import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $agentSidebarChoice } from '@/store/simplified-agent-mode'

import { SettingsView } from './entry'

vi.mock('../settings', () => ({ SettingsView: () => <div>Original settings</div> }))
vi.mock('./index', () => ({
  AgentSettingsView: ({ owner }: { owner: { connectionId?: string; profile: string } }) => (
    <div>
      {owner.connectionId}:{owner.profile}
    </div>
  )
}))
afterEach(() => {
  cleanup()
  $agentSidebarChoice.set(false)
})

it('keeps the original screen when v2 is off and gates an existing agent URL immediately', async () => {
  $agentSidebarChoice.set(false)
  render(
    <MemoryRouter initialEntries={['/settings?agent=research&connection=beta']}>
      <SettingsView onClose={() => {}} />
    </MemoryRouter>
  )
  expect(screen.getByText('Original settings')).toBeTruthy()
  act(() => $agentSidebarChoice.set(true))
  expect(await screen.findByText('beta:research')).toBeTruthy()
  act(() => $agentSidebarChoice.set(false))
  expect(screen.getByText('Original settings')).toBeTruthy()
  expect(screen.queryByText('beta:research')).toBeNull()
})
