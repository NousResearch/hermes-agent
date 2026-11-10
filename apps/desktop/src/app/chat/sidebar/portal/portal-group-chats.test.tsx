import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { RootTooltipProvider } from '@/components/ui/tooltip'
import type * as GroupRegistry from '@/plugins/hermes-bots/canonical-group-registry'
import { useGatewayGroupRooms } from '@/plugins/hermes-bots/canonical-group-registry'
import { $botMeta, $lastRoster } from '@/plugins/hermes-bots/data'
import { $groupChats } from '@/plugins/hermes-bots/group-chat'
import { openGroupChat } from '@/plugins/hermes-bots/group-chat-view'
import type * as BotsI18n from '@/plugins/hermes-bots/i18n'

import { PortalGroupChats } from './portal-group-chats'

vi.mock('@/plugins/hermes-bots/canonical-group-registry', async original => ({
  ...await original<typeof GroupRegistry>(),
  useGatewayGroupRooms: vi.fn(() => [])
}))
vi.mock('@/plugins/hermes-bots/group-chat-view', () => ({ openGroupChat: vi.fn() }))
vi.mock('@/plugins/hermes-bots/i18n', async original => {
  const module = await original<typeof BotsI18n>()

  return { ...module, useBots: () => module.BOTS_LOCALES.en }
})

afterEach(() => { cleanup(); $botMeta.set({}); $lastRoster.set([]); $groupChats.set({}); vi.clearAllMocks() })

it('opens same-named rooms using their exact gateway authority and retains desktop rooms', () => {
  const rooms = ['alpha', 'beta'].map(connectionId => ({
    binding: { connectionId, profile: 'default', roomId: 'room', roomName: 'Team' },
    key: `canonical:${connectionId}:default:room`, name: 'Team', members: [], updatedAt: 0,
    room: { room_id: 'room', name: 'Team', members: [] }
  }))

  vi.mocked(useGatewayGroupRooms).mockReturnValue(rooms)
  $groupChats.set({ 'Across gateways': { log: [], watermarks: {}, roomId: 'desktop-room', engine: 'desktop', members: [
    { name: 'one', connectionId: 'alpha', sourceScoped: true },
    { name: 'two', connectionId: 'beta', sourceScoped: true }
  ] } })
  render(<RootTooltipProvider><PortalGroupChats query="" /></RootTooltipProvider>)
  const groups = screen.getAllByRole('button', { name: 'Team' })
  fireEvent.click(groups[1])
  expect(openGroupChat).toHaveBeenLastCalledWith('canonical:beta:default:room', 'key')
  fireEvent.click(groups[0])
  expect(openGroupChat).toHaveBeenLastCalledWith('canonical:alpha:default:room', 'key')
  fireEvent.click(screen.getByRole('button', { name: /Across gateways/ }))
  expect(openGroupChat).toHaveBeenLastCalledWith('Across gateways', 'name')
  expect(screen.getByRole('button', { name: 'New group chat' })).toBeTruthy()
})
