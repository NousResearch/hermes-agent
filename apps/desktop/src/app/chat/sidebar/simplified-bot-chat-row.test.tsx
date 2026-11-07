import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $openBotChat, $pendingBotOpen } from '@/plugins/hermes-bots/bot-state'
import { openRosterBot } from '@/plugins/hermes-bots/roster-actions'
import { botWorkspaceOwnerKey } from '@/plugins/hermes-bots/routing'
import { $selectedStoredSessionId } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { SimplifiedBotChatRow } from './simplified-bot-chat-row'

vi.mock('@/plugins/hermes-bots/roster-actions', () => ({ openRosterBot: vi.fn(async () => true) }))

const agent = {
  id: 'remote-alpha',
  profile: 'alpha',
  label: 'Alpha',
  gateway: 'Work',
  connectionId: 'remote',
  connectionKind: 'ssh' as const,
  targetProfile: 'actual-alpha'
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $sessionTiles.set([])
  $openBotChat.set(null)
  $pendingBotOpen.set(null)
  $selectedStoredSessionId.set(null)
})

it('only opens on click and preserves the exact gateway and target profile', () => {
  render(<SimplifiedBotChatRow agent={agent} />)
  expect(openRosterBot).not.toHaveBeenCalled()
  const row = screen.getByRole('button', { name: 'Bot Chat' })
  expect(row.querySelector('.codicon-robot')).not.toBeNull()
  fireEvent.click(row)
  expect(openRosterBot).toHaveBeenCalledWith(
    expect.objectContaining({
      name: 'alpha',
      connectionId: 'remote',
      connectionKind: 'ssh',
      targetProfile: 'actual-alpha',
      sourceScoped: true
    })
  )
})

it('shows progress and selection only on the matching owner and focused canonical chat', () => {
  render(
    <>
      <SimplifiedBotChatRow agent={agent} />
      <SimplifiedBotChatRow agent={{ ...agent, connectionId: 'other' }} />
    </>
  )
  const [row, other] = screen.getAllByRole('button', { name: 'Bot Chat' })
  act(() => $pendingBotOpen.set({ generation: 1, key: 'remote::alpha' }))
  expect(row.getAttribute('aria-busy')).toBe('true')
  expect(other.hasAttribute('disabled')).toBe(false)
  fireEvent.click(row)
  expect(openRosterBot).not.toHaveBeenCalled()
  act(() => {
    $pendingBotOpen.set(null)
    $selectedStoredSessionId.set('tip')
    $openBotChat.set({ key: 'remote::alpha', openedRegistryId: 'root', openedSessionId: 'tip' })
  })
  expect(row.getAttribute('aria-current')).toBe('page')
  expect(other.hasAttribute('aria-current')).toBe(false)
  act(() => $selectedStoredSessionId.set('side-chat'))
  expect(row.hasAttribute('aria-current')).toBe(false)
})

it('highlights a canonical tab after its temporary open claim is released', () => {
  $sessionTiles.set([
    {
      storedSessionId: 'canonical-tip',
      workspaceMode: 'bots',
      workspaceOwnerKey: botWorkspaceOwnerKey({ name: 'alpha', connectionId: 'remote', sourceScoped: true }),
      workspaceTabTitle: 'Bot Chat'
    }
  ])
  render(<SimplifiedBotChatRow agent={agent} />)
  act(() => $selectedStoredSessionId.set('canonical-tip'))
  expect(screen.getByRole('button', { name: 'Bot Chat' }).getAttribute('aria-current')).toBe('page')
  act(() => $selectedStoredSessionId.set('other'))
  expect(screen.getByRole('button', { name: 'Bot Chat' }).hasAttribute('aria-current')).toBe(false)
  act(() => $selectedStoredSessionId.set('canonical-tip'))
  expect(screen.getByRole('button', { name: 'Bot Chat' }).getAttribute('aria-current')).toBe('page')
})
