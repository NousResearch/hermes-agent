import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $openBotChat, $pendingBotOpen } from '@/plugins/hermes-bots/bot-state'
import { openRosterBot } from '@/plugins/hermes-bots/roster-actions'
import { $selectedStoredSessionId } from '@/store/session'

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
