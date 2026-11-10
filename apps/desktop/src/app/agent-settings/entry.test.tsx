import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $agentSidebarChoice } from '@/store/simplified-agent-mode'

import { SettingsView } from './entry'

vi.mock('../settings', () => ({ SettingsView: () => <div>Original settings</div> }))
vi.mock('./index', () => ({
  AgentSettingsView: ({ owner, classic }: { owner: { connectionId?: string; profile: string }; classic: boolean }) => (
    <div>
      {owner.connectionId}:{owner.profile}:{classic ? 'classic' : 'redesigned'}
    </div>
  )
}))
afterEach(() => {
  cleanup()
  $agentSidebarChoice.set(false)
})

it('keeps an explicit owner pinned across presentation changes', async () => {
  $agentSidebarChoice.set(false)
  render(
    <MemoryRouter initialEntries={['/settings?agent=research&connection=beta']}>
      <SettingsView onClose={() => {}} />
    </MemoryRouter>
  )
  expect(await screen.findByText('beta:research:classic')).toBeTruthy()
  act(() => $agentSidebarChoice.set(true))
  expect(await screen.findByText('beta:research:redesigned')).toBeTruthy()
  act(() => $agentSidebarChoice.set(false))
  expect(screen.getByText('beta:research:classic')).toBeTruthy()
  expect(screen.queryByText('Original settings')).toBeNull()
})
