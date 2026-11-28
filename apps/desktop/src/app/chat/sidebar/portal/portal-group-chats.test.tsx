import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { RootTooltipProvider } from '@/components/ui/tooltip'
import type * as GroupRegistry from '@/plugins/hermes-bots/canonical-group-registry'
import { useGatewayGroupRooms } from '@/plugins/hermes-bots/canonical-group-registry'
import { $botMeta, $lastRoster } from '@/plugins/hermes-bots/data'
import { $groupChats } from '@/plugins/hermes-bots/group-chat'
import { openGroupChat } from '@/plugins/hermes-bots/group-chat-view'
import type * as BotsI18n from '@/plugins/hermes-bots/i18n'
import { wipeSessionListsForGatewaySwitch } from '@/store/gateway-switch'
import { $activeGatewayProfile } from '@/store/profile'
import { $seenGroupRooms } from '@/store/seen-group-rooms'
import { $connection } from '@/store/session'

import { PortalGroupChats, PortalNewGroupChatAction, useGroupChatMatches } from './portal-group-chats'

function GroupChats() {
  return <PortalGroupChats matches={useGroupChatMatches('')} />
}

vi.mock('@/plugins/hermes-bots/canonical-group-registry', async original => ({
  ...await original<typeof GroupRegistry>(),
  useGatewayGroupRooms: vi.fn(() => [])
}))
vi.mock('@/plugins/hermes-bots/group-chat-view', () => ({ openGroupChat: vi.fn() }))

const dialog = vi.hoisted(() => ({ props: null as null | { onClose: () => void; onCreated: (key: string, origin: 'key') => void } }))

vi.mock('@/plugins/hermes-bots/create-dialog', () => ({
  CreateGroupChatDialog: (props: NonNullable<typeof dialog.props>) => {
    dialog.props = props

    return <div role="dialog" />
  }
}))
vi.mock('@/plugins/hermes-bots/i18n', async original => {
  const module = await original<typeof BotsI18n>()

  return { ...module, useBots: () => module.BOTS_LOCALES.en }
})

afterEach(() => { cleanup(); $botMeta.set({}); $lastRoster.set([]); $groupChats.set({}); $seenGroupRooms.set([]); vi.clearAllMocks() })

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
  render(<RootTooltipProvider><GroupChats /></RootTooltipProvider>)
  const groups = screen.getAllByRole('button', { name: 'Team' })
  fireEvent.click(groups[1])
  expect(openGroupChat).toHaveBeenLastCalledWith('canonical:beta:default:room', 'key')
  fireEvent.click(groups[0])
  expect(openGroupChat).toHaveBeenLastCalledWith('canonical:alpha:default:room', 'key')
  fireEvent.click(screen.getByRole('button', { name: /Across gateways/ }))
  expect(openGroupChat).toHaveBeenLastCalledWith('Across gateways', 'name')
  expect(screen.getByRole('button', { name: 'New group chat' })).toBeTruthy()
})

it('counts members from the published roster', () => {
  $lastRoster.set(['one', 'two'].map(name => ({ name, connectionId: 'local' })))
  $groupChats.set({ Pair: { log: [], watermarks: {}, roomId: 'pair', engine: 'desktop', members: [
    { name: 'one', connectionId: 'local', sourceScoped: true },
    { name: 'two', connectionId: 'local', sourceScoped: true }
  ] } })
  render(<RootTooltipProvider><GroupChats /></RootTooltipProvider>)
  expect(screen.getByRole('button', { name: /Pair/ }).textContent).toContain('2')
})

it('leaves no section behind while there is nothing to list (HADESK-132)', () => {
  vi.mocked(useGatewayGroupRooms).mockReturnValue([])
  render(<RootTooltipProvider><GroupChats /></RootTooltipProvider>)
  expect(screen.queryByText('Group chats')).toBeNull()
  expect(screen.queryByRole('button', { name: 'New group chat' })).toBeNull()
})

const nextFrame = () => act(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())))

it('a dismissed group dialog hands focus back to its "+"; a created group chat keeps it', async () => {
  render(<RootTooltipProvider><PortalNewGroupChatAction /></RootTooltipProvider>)
  const plus = screen.getByRole('button', { name: 'New group chat' })

  fireEvent.click(plus)
  ;(window.document.activeElement as HTMLElement | null)?.blur()
  act(() => dialog.props?.onClose())
  await nextFrame()
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(window.document.activeElement).toBe(plus)

  fireEvent.click(plus)
  plus.blur()
  act(() => {
    dialog.props?.onClose()
    dialog.props?.onCreated('canonical:local:default:room', 'key')
  })
  await nextFrame()
  expect(openGroupChat).toHaveBeenLastCalledWith('canonical:local:default:room', 'key')
  expect(window.document.activeElement).not.toBe(plus)
})

// The hook lists the active agent's rooms only (a room lives in the store of the agent that made it);
// the section is not one agent's, so it keeps the rooms it has seen on this machine.
describe('rooms seen under an agent that was active earlier', () => {
  const roomOf = (profile: string, roomId: string, name: string) => ({
    binding: { connectionId: 'local', profile, roomId, roomName: name },
    key: `canonical:local:${profile}:${roomId}`, name, members: [], updatedAt: 0,
    room: { room_id: roomId, name, members: [] }
  })

  function Names() {
    return <p>{useGroupChatMatches('').rooms.map(room => `${room.name}@${room.binding.profile}`).join(', ') || 'none'}</p>
  }

  const listed = () => screen.getByRole('paragraph').textContent

  function show(profile: string, rooms: ReturnType<typeof roomOf>[], view?: ReturnType<typeof render>) {
    vi.mocked(useGatewayGroupRooms).mockReturnValue(rooms)
    act(() => $activeGatewayProfile.set(profile))

    return view ? (view.rerender(<Names />), view) : render(<Names />)
  }

  afterEach(() => {
    $activeGatewayProfile.set('default')
    $connection.set(null)
  })

  it('stay listed when another agent becomes the active one', () => {
    $connection.set({ connectionId: 'local' } as never)
    const view = show('default', [roomOf('default', 'r1', 'Launch crew')])
    expect(listed()).toBe('Launch crew@default')

    show('ops', [], view)
    expect(listed()).toBe('Launch crew@default')

    show('ops', [roomOf('ops', 'r2', 'Field notes')], view)
    expect(listed()).toBe('Launch crew@default, Field notes@ops')
  })

  it('are still listed when the sidebar is closed and opened again under another agent', () => {
    $connection.set({ connectionId: 'local' } as never)
    show('default', [roomOf('default', 'r1', 'Launch crew')]).unmount()

    show('ops', [])
    expect(listed()).toBe('Launch crew@default')
  })

  it('are dropped when the window is re-homed to another backend', () => {
    $connection.set({ connectionId: 'local' } as never)
    const view = show('default', [roomOf('default', 'r1', 'Launch crew')])

    act(() => wipeSessionListsForGatewaySwitch())
    show('ops', [], view)
    expect(listed()).toBe('none')
  })

  it('leave when their own agent no longer lists them', () => {
    $connection.set({ connectionId: 'local' } as never)
    const view = show('default', [roomOf('default', 'r1', 'Launch crew')])
    show('ops', [], view)

    show('default', [], view)
    expect(listed()).toBe('none')
  })

  it('are one row each on a backend whose agents all list the same rooms', () => {
    $connection.set({ connectionId: 'local' } as never)
    const view = show('default', [roomOf('default', 'r1', 'Launch crew')])

    show('ops', [roomOf('ops', 'r1', 'Launch crew')], view)
    expect(listed()).toBe('Launch crew@ops')
  })

  it('leave with an agent that is no longer in the roster', () => {
    $connection.set({ connectionId: 'local' } as never)
    $lastRoster.set([{ connectionId: 'local', name: 'default' }, { connectionId: 'local', name: 'ops' }] as never)
    const view = show('default', [roomOf('default', 'r1', 'Launch crew')])
    show('ops', [], view)
    expect(listed()).toBe('Launch crew@default')

    // An agent of the same name on another machine is not this room's agent.
    act(() =>
      $lastRoster.set([{ connectionId: 'local', name: 'ops' }, { connectionId: 'elsewhere', name: 'default' }] as never)
    )
    expect(listed()).toBe('none')
  })

  it('keep their place when the active agent changes', () => {
    $connection.set({ connectionId: 'local' } as never)
    const crew = roomOf('default', 'r1', 'Launch crew')
    const notes = { ...roomOf('ops', 'r2', 'Field notes'), updatedAt: 5 }
    const view = show('default', [crew])
    show('ops', [notes], view)
    const order = listed()

    show('default', [crew], view)
    expect(listed()).toBe(order)
    expect(order).toBe('Field notes@ops, Launch crew@default')
  })

  it('are forgotten on another machine', () => {
    $connection.set({ connectionId: 'local' } as never)
    const view = show('default', [roomOf('default', 'r1', 'Launch crew')])

    // The other machine's own list arrives with the switch; nothing of this one's rides along.
    vi.mocked(useGatewayGroupRooms).mockReturnValue([])
    act(() => $connection.set({ connectionId: 'elsewhere' } as never))
    show('ops', [], view)
    expect(listed()).toBe('none')
  })
})
