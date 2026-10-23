import type * as HermesSdk from '@hermes/plugin-sdk'
import { host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { $canonicalGroupBindings } from './canonical-group-registry'
import { canonicalGroupCreateError } from './canonical-groups'
import { CreateGroupChatDialog } from './create-dialog'
import { $botMeta } from './data'
import { $groupChats, $groupChatWorkspace, updateGroupChat } from './group-chat'
import type * as GroupChatModule from './group-chat'
import type * as GroupChatParts from './group-chat-parts'
import { GroupChatWorkspace } from './group-chat-view'
import { botsText } from './i18n'
import { translateBots } from './i18n-test-helper'
import { renderRosterToolbar } from './roster-pane-toolbar'
import type { RosterRow } from './types'

const { request, notify, openWorkspace, activation } = vi.hoisted(() => ({ request: vi.fn(), notify: vi.fn(), openWorkspace: vi.fn(), activation: { epoch: 1 } }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { captureGroupRequests } = await import('./canonical-group-contract-test-utils')
  const { en } = await import('@/i18n/en')

  return {
    ...sdk,
    gatewayActivationEpoch: () => activation.epoch,
    host: {
      ...sdk.host, requestProfile: captureGroupRequests(request).request, notify, openWorkspace,
      request: (method: string, params?: Record<string, unknown>) => request(null, method, params),
      connections: vi.fn(async () => []),
      state: {
        ...sdk.host.state,
        connectionId: sdk.atom<string | null>('local'),
        profile: sdk.atom('default'),
        gateway: sdk.atom('open')
      }
    },
    useI18n: () => ({ locale: 'en', t: en }),
    usePluginI18n: () => translateBots
  }
})
vi.mock('./group-chat', async importOriginal => {
  const actual = await importOriginal<typeof GroupChatModule>()

  return {
    ...actual,
    // Keep actual local creation, but do not schedule the unrelated remote mirror.
    updateGroupChat: vi.fn((group, mutate) => actual.updateGroupChat(group, mutate, { sync: false }))
  }
})
vi.mock('./group-chat-parts', async importOriginal => ({
  ...await importOriginal<typeof GroupChatParts>(),
  // Avatar generation is unrelated to the capability decision; never request a model.
  GroupImageControls: () => null
}))

const state = {
  connectionId: host.state.connectionId as WritableAtom<string | null>,
  profile: host.state.profile as WritableAtom<string>,
  gateway: host.state.gateway as WritableAtom<string>
}

const roster: RosterRow[] = [{ name: 'alpha', connectionId: 'local' }, { name: 'beta', connectionId: 'local' }]

const mixedRoster: RosterRow[] = [
  ...roster,
  { name: 'remote', connectionId: 'other-box', connectionLabel: 'Other box', remoteSource: true },
  {
    name: 'spoofed',
    connectionId: 'local',
    remoteSource: true,
    route: { connectionId: 'other-box', mode: 'remote', profile: 'spoofed', targetProfile: 'spoofed' }
  }
]

const unavailable = CANONICAL_GROUP_LOCALES.en.driverUnavailable

// Decision-relevant fields emitted by the actual canonical capabilities producer.
const canonicalUnavailable = {
  driver: false, persistent_process: true, features: ['room_identity', 'monotonic_log', 'replayable_disband']
}

// App-managed hosted capabilities keep their protocol/authority when the driver stops.
// Their RoomLink catalog deliberately reports persistent_process:false.
const appManagedUnavailable = {
  driver: false, persistent_process: false, protocol_version: 2,
  authority_gateway_id: 'installation:app-managed',
  features: ['authority_epoch', 'coordinator_fencing', 'room_identity', 'monotonic_log'],
  methods: ['groups.capabilities', 'groups.create', 'groups.state', 'groups.send']
}

const legacy = { driver: false, persistent_process: false }

// As on main, a server without the group surface (-32601) never runs classic rooms: it is too old for group chats.
const noGroupSurface = Object.assign(new Error('unknown method: groups.capabilities'), { code: -32601 })

const refused = [canonicalUnavailable, appManagedUnavailable, { ...legacy, methods: ['groups.create'] }, { driver: 'true', persistent_process: false }, null,
  noGroupSurface]

beforeEach(() => {
  activation.epoch = 1
  state.connectionId.set('local')
  state.profile.set('default')
  state.gateway.set('open')
  $canonicalGroupBindings.set({})
  $groupChats.set({})
  $groupChatWorkspace.set(null)
  $botMeta.set({})
  request.mockReset()
  notify.mockReset()
  openWorkspace.mockReset().mockReturnValue(() => undefined)
  vi.mocked(updateGroupChat).mockClear()
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})
afterEach(() => {
  cleanup()
  $groupChats.set({})
  localStorage.clear()
  vi.restoreAllMocks()
})

function answer(capabilities: unknown) {
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {
      if (capabilities instanceof Error) {throw capabilities}

      return capabilities
    }

    if (method === 'groups.create') {return { room: { room_id: params.room_id, name: params.name, members: params.members } }}

    if (method === 'profiles.configure') {return {}}
    throw new Error(`Unexpected RPC: ${method}`)
  })
}

async function submitDialog() {
  const onCreated = vi.fn()
  const onClose = vi.fn()
  render(<CreateGroupChatDialog onClose={onClose} onCreated={onCreated} open roster={roster} />)

  for (const checkbox of screen.getAllByRole('checkbox').slice(0, 3)) {fireEvent.click(checkbox)}
  await waitFor(() => expect(request.mock.calls.map(call => call[1])).toContain('groups.capabilities'))
  await act(async () => {})
  const submit = screen.getByRole('button', { name: 'Create Group (2)' }) as HTMLButtonElement

  if (!submit.disabled) {
    await act(async () => { fireEvent.click(submit) })
  }

  return { onCreated, onClose }
}

function pendingCreation() {
  let finish!: () => void
  const serverRooms = new Map<string, unknown>()
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {return { driver: true, persistent_process: true }}

    // A legacy room with no gateway room of its name yet (#140805 lookup answers empty).
    if (method === 'groups.list') {return { rooms: [], next_offset: null }}

    if (method === 'groups.create') {
      const room = { room_id: params.room_id, name: params.name, members: params.members }
      serverRooms.set(room.room_id, room)

      return new Promise(resolve => { finish = () => resolve({ room }) })
    }

    throw new Error(`Unexpected RPC: ${method}`)
  })

  return { serverRooms, finish: () => finish() }
}

it.each(refused)('classifies %j as unavailable on both surfaces: no legacy renderer, no legacy creation', async value => {
  answer(value)
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })

  // A gateway whose group driver isn't running, or no group surface or an answer this Desktop can't read (too old).
  const words = (value as { driver?: unknown } | null)?.driver === false ? CANONICAL_GROUP_LOCALES.en.groupDriverStopped
    : CANONICAL_GROUP_LOCALES.en.groupsUnsupported

  expect(screen.getByText(words)).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  expect(screen.queryByRole('textbox')).toBeNull()
  expect(screen.queryByRole('button', { name: 'Start new group' })).toBeNull()
  cleanup()

  const { onCreated, onClose } = await submitDialog()
  expect(screen.getByRole('status').textContent).toBe(unavailable)
  expect((screen.getByRole('button', { name: 'Create Group (2)' }) as HTMLButtonElement).disabled).toBe(true)
  expect(notify).not.toHaveBeenCalled()
  expect(onCreated).not.toHaveBeenCalled()
  expect(onClose).not.toHaveBeenCalled()
  expect(updateGroupChat).not.toHaveBeenCalled()
  expect($groupChats.get()).toEqual({})
  expect(request.mock.calls.map(call => call[1])).toEqual(['groups.capabilities', 'groups.capabilities'])
})

it('keeps positive classifications working: legacy renders and creates locally, canonical creates a gateway room', async () => {
  answer(legacy)
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  expect(screen.getByRole('textbox')).toBeTruthy()
  cleanup()
  expect((await submitDialog()).onCreated).toHaveBeenCalledOnce()
  expect(updateGroupChat).toHaveBeenCalledOnce()
  cleanup()

  answer({ driver: true, persistent_process: true })
  const { onCreated } = await submitDialog()
  expect(onCreated).toHaveBeenCalledOnce()
  expect(Object.values($canonicalGroupBindings.get())).toHaveLength(1)
  expect(updateGroupChat).toHaveBeenCalledOnce()
  expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(1)
})

it('canonical creation offers only members on the current connection before submission', async () => {
  answer({ driver: true, persistent_process: true })
  const { baseElement } = render(<CreateGroupChatDialog onClose={vi.fn()} open roster={mixedRoster} />)

  expect(baseElement.querySelector('[data-bot-face="remote"]')).toBeNull()
  expect(baseElement.querySelector('[data-bot-face="spoofed"]')).toBeNull()
  await waitFor(() => {
    expect(request.mock.calls.map(call => call[1])).toContain('groups.capabilities')
    expect(baseElement.querySelector('[data-bot-face="alpha"]')).not.toBeNull()
    expect(baseElement.querySelector('[data-bot-face="beta"]')).not.toBeNull()
  })

  for (const checkbox of screen.getAllByRole('checkbox').slice(0, 2)) {fireEvent.click(checkbox)}
  const submit = screen.getByRole('button', { name: 'Create Group (2)' }) as HTMLButtonElement
  await waitFor(() => expect(submit.disabled).toBe(false))
  expect(baseElement.querySelector('[data-bot-face="remote"]')).toBeNull()
  expect(baseElement.querySelector('[data-bot-face="spoofed"]')).toBeNull()
})

it('canonical creation offers only the bots the authority says this caller can seat (#141178)', async () => {
  // `parked` is on this connection but not served (or a sibling this socket may
  // not seat): groups.create would refuse it, so the picker must not offer it.
  const withParked: RosterRow[] = [...roster, { name: 'parked', connectionId: 'local' }]
  answer({ driver: true, persistent_process: true, seatable_profiles: ['alpha', 'beta', 'default'] })
  const { baseElement } = render(<CreateGroupChatDialog onClose={vi.fn()} open roster={withParked} />)

  await waitFor(() => {
    expect(baseElement.querySelector('[data-bot-face="alpha"]')).not.toBeNull()
    expect(baseElement.querySelector('[data-bot-face="beta"]')).not.toBeNull()
  })
  expect(baseElement.querySelector('[data-bot-face="parked"]')).toBeNull()
  expect(screen.getAllByRole('checkbox')).toHaveLength(2)
})

it('rechecks the current mode before creating after a mixed legacy selection', async () => {
  let capabilityReads = 0
  const onCreated = vi.fn()
  const onClose = vi.fn()
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {
      capabilityReads += 1

      return capabilityReads === 1 ? legacy : { driver: true, persistent_process: true }
    }

    if (method === 'groups.create') {
      return { room: { room_id: params.room_id, name: params.name, members: params.members } }
    }

    throw new Error(`Unexpected RPC: ${method}`)
  })

  const { baseElement } = render(<CreateGroupChatDialog onClose={onClose} onCreated={onCreated} open roster={mixedRoster} />)
  await waitFor(() => expect(capabilityReads).toBe(1))
  await waitFor(() => {
    expect(baseElement.querySelector('[data-bot-face="remote"]')).not.toBeNull()
    expect(baseElement.querySelector('[data-bot-face="spoofed"]')).not.toBeNull()
  })

  for (const checkbox of screen.getAllByRole('checkbox').slice(0, 3)) {fireEvent.click(checkbox)}
  const submit = screen.getByRole('button', { name: 'Create Group (3)' })
  expect((submit as HTMLButtonElement).disabled).toBe(false)

  await act(async () => { fireEvent.click(submit) })
  await waitFor(() => expect(notify).toHaveBeenCalledWith({ kind: 'error', message: unavailable }))
  expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(0)
  expect(onCreated).not.toHaveBeenCalled()
  expect(onClose).not.toHaveBeenCalled()
})

it.each(['profile', 'gateway', 'same-route-activation'] as const)('rejects a %s change queued after fresh capability validation', async kind => {
  let capabilityReads = 0
  const onCreated = vi.fn()
  const onClose = vi.fn()
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {
      capabilityReads += 1

      if (capabilityReads === 2) {
        queueMicrotask(() => queueMicrotask(() => queueMicrotask(() => queueMicrotask(() => moveSource(kind)))))
      }

      return { driver: true, persistent_process: true }
    }

    if (method === 'groups.create') {
      return { room: { room_id: params.room_id, name: params.name, members: params.members } }
    }

    throw new Error(`Unexpected RPC: ${method}`)
  })

  render(<CreateGroupChatDialog onClose={onClose} onCreated={onCreated} open roster={roster} />)
  await waitFor(() => expect(capabilityReads).toBe(1))

  for (const checkbox of screen.getAllByRole('checkbox').slice(0, 2)) {fireEvent.click(checkbox)}
  const submit = screen.getByRole('button', { name: 'Create Group (2)' })
  await waitFor(() => expect((submit as HTMLButtonElement).disabled).toBe(false))

  await act(async () => { fireEvent.click(submit) })
  await waitFor(() => expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(0))
  await waitFor(() => expect(notify).toHaveBeenCalledWith({ kind: 'error', message: unavailable }))
  expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(0)
  expect(updateGroupChat).not.toHaveBeenCalled()
  expect(onCreated).not.toHaveBeenCalled()
  expect(onClose).not.toHaveBeenCalled()
})

it.each(['profile', 'gateway', 'same-route-activation'] as const)('does not persist a legacy group after a %s change queued after fresh capability validation', async kind => {
  let capabilityReads = 0
  const onCreated = vi.fn()
  const onClose = vi.fn()
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {
      capabilityReads += 1

      if (capabilityReads === 2) {
        queueMicrotask(() => queueMicrotask(() => queueMicrotask(() => queueMicrotask(() => moveSource(kind)))))
      }

      return legacy
    }

    if (method === 'profiles.configure') {return {}}

    if (method === 'groups.create') {
      return { room: { room_id: params.room_id, name: params.name, members: params.members } }
    }

    throw new Error(`Unexpected RPC: ${method}`)
  })

  render(<CreateGroupChatDialog onClose={onClose} onCreated={onCreated} open roster={roster} />)
  await waitFor(() => expect(capabilityReads).toBe(1))

  for (const checkbox of screen.getAllByRole('checkbox').slice(0, 2)) {fireEvent.click(checkbox)}
  const submit = screen.getByRole('button', { name: 'Create Group (2)' })
  await waitFor(() => expect((submit as HTMLButtonElement).disabled).toBe(false))

  await act(async () => { fireEvent.click(submit) })
  await act(async () => {})
  expect(request.mock.calls.filter(call => call[1] === 'profiles.configure')).toHaveLength(0)
  expect(updateGroupChat).not.toHaveBeenCalled()
  expect(notify).toHaveBeenCalledWith({ kind: 'error', message: unavailable })
  expect(onCreated).not.toHaveBeenCalled()
  expect(onClose).not.toHaveBeenCalled()
})

function moveSource(kind: 'profile' | 'gateway' | 'same-route-activation') {
  if (kind === 'profile') {state.profile.set('other')}

  if (kind === 'gateway') {state.gateway.set('closed')}

  if (kind === 'same-route-activation') {
    activation.epoch++
    state.profile.set('default')
  }
}

it.each(['profile', 'gateway', 'same-route-activation'] as const)('dialog: a creation approved before the %s moved is kept on its owner and never published', async kind => {
  const pending = pendingCreation()
  const { onCreated, onClose } = await submitDialog()
  expect(pending.serverRooms.size).toBe(1)
  await act(async () => { moveSource(kind); pending.finish() })
  expect($canonicalGroupBindings.get()).toEqual({})
  expect(onCreated).not.toHaveBeenCalled()
  expect(onClose).not.toHaveBeenCalled()
  expect(pending.serverRooms.size).toBe(1)
  expect(request.mock.calls[1][0]).toMatchObject({ connectionId: 'local', profile: 'default' })
})

it.each(['profile', 'gateway', 'same-route-activation'] as const)('workspace: a capability read before the %s moved neither creates nor opens a room', async kind => {
  const pending = pendingCreation()
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  const button = screen.getByRole('button', { name: 'Start new group' })
  expect((button as HTMLButtonElement).disabled).toBe(false)
  // Click after the source moved but before React re-renders: the stale capability must not create.
  await act(async () => { moveSource(kind); fireEvent.click(button) })
  expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(0)
  expect(pending.serverRooms.size).toBe(0)
  expect(openWorkspace).not.toHaveBeenCalled()
})

// The New… menu opens New Group Chat. It is offered only when the dialog can
// seat two bots by the SAME rule the dialog's picker uses; counting the whole
// roster opened a dialog with one checkbox and a Create button that never woke.
it.each([
  ['one seatable bot', ['alpha', 'default'], true],
  ['two seatable bots', ['alpha', 'beta'], false]
])('New Group Chat in the New… menu, with %s', async (_label, seatable, disabled) => {
  const withParked: RosterRow[] = [...roster, { name: 'parked', connectionId: 'local' }]
  answer({ driver: true, persistent_process: true, seatable_profiles: seatable })
  const setGroupCreateOpen = vi.fn()

  render(
    renderRosterToolbar({
      activeFilterCount: 0,
      activeSourceRoster: withParked,
      activityFilter: 'all',
      activityToasts: false,
      b: botsText(),
      gatewayFilter: 'all',
      gatewayOptions: [],
      query: '',
      roster: withParked,
      rowKindFilter: 'all',
      setActivityFilter: vi.fn(),
      setCreateOpen: vi.fn(),
      setGatewayFilter: vi.fn(),
      setGroupCreateOpen,
      setQuery: vi.fn(),
      setRowKindFilter: vi.fn(),
      setSectionDialog: vi.fn(),
      showRosterFilters: false,
      showRosterSearch: false,
      showRosterTools: false
    })
  )

  fireEvent.keyDown(screen.getByRole('button', { name: botsText().roster.newBotOrGroup }), { key: 'Enter' })
  const item = await screen.findByRole('menuitem', { name: botsText().group.newTitle })

  // Settles once the authority's seatable roster is in: all three bots are on
  // this connection, but only the ones it will seat count.
  await waitFor(() => expect(item.hasAttribute('data-disabled')).toBe(disabled))
  fireEvent.click(item)
  expect(setGroupCreateOpen).toHaveBeenCalledTimes(disabled ? 0 : 1)
})

const mini = [{ name: 'mini-a', connectionId: 'mini', connectionLabel: 'Mini', remoteSource: true },
  { name: 'mini-b', connectionId: 'mini', connectionLabel: 'Mini', remoteSource: true }] as RosterRow[]

/** A classic room `group` on this gateway connection, which has no gateway room of that name. */
function classicRoom(group: string, members: RosterRow[], room: Record<string, unknown> = {}, lookup?: Error) {
  $groupChats.set({ [group]: { log: [], running: false, members, ...room } } as never)
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {return { driver: true, persistent_process: true }}

    if (method === 'groups.list') {
      if (lookup) {throw lookup}

      return { rooms: [], next_offset: null }
    }

    throw new Error(`Unexpected RPC: ${method}`)
  })
}

// On a gateway connection the gateway owns every group chat, as on main: a classic room never runs
// here, whichever computers its Bots are on. The gate says why and what to do, and only Bots that
// can form a gateway group here are offered Start new group.
it.each([
  ['Bots that can form a gateway group here', roster, 'legacyRoom', '', true],
  ['Bots on another connection', mini, 'legacyRoomElsewhere', 'Bots on other connections: Mini', false],
  ['a Bot here and one elsewhere', [roster[0], mini[0]], 'legacyRoomElsewhere', 'Bots on other connections: Mini', false],
  ['Bots here that cannot form a gateway group as they are', [roster[0]], 'legacyRoomBlocked', CANONICAL_GROUP_LOCALES.en.createGroupSize, false]
] as const)('keeps a classic room with %s on the gate and says what to do', async (_case, members, message, problem, start) => {
  classicRoom('Classic', [...members])
  await act(async () => { render(<GroupChatWorkspace group="Classic" members={[...members]} />) })
  await screen.findByText(CANONICAL_GROUP_LOCALES.en[message])
  expect(screen.queryByRole('status')?.textContent ?? '').toBe(problem)
  expect(screen.queryByRole('textbox')).toBeNull()
  expect(Boolean(screen.queryByRole('button', { name: 'Start new group' }))).toBe(start)
  expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
})

it('shows a classic room\'s earlier messages on the gate read only and inert, and a failed lookup with Retry', async () => {
  const log = [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'u1', text: 'Show [the docs](https://example.com/docs)' },
    { at: 2, from: { kind: 'member', name: 'mini-a', source: 'Mini' }, id: 'm1',
      text: 'MEDIA:/home/user/.ssh/id_rsa\n\n![shot](/tmp/shot.png) [panel](http://127.0.0.1:9119/)' }]

  // Even a room this Desktop already ran on `mini` (local plumbing sessions) stays on the gate.
  classicRoom('Remote', mini, { log, sessions: { 'thread:t1::mini::mini-a': 'stored-1' } })
  const view = render(<GroupChatWorkspace group="Remote" members={mini} />)
  await screen.findByText(CANONICAL_GROUP_LOCALES.en.legacyRoomElsewhere)
  // Shown, read only, with nothing resolved and only the public link live.
  const earlier = screen.getByRole('region', { name: CANONICAL_GROUP_LOCALES.en.earlierMessages })
  expect(earlier.textContent).toContain('MEDIA:/home/user/.ssh/id_rsa')
  expect(earlier.querySelector('img, video, audio, button')).toBeNull()
  expect([...earlier.querySelectorAll('a')].map(link => link.getAttribute('href'))).toEqual(['https://example.com/docs'])
  // No media or file read reached any gateway: only the capability check and the room lookup.
  expect(new Set(request.mock.calls.map(call => call[1]))).toEqual(new Set(['groups.capabilities', 'groups.list']))
  view.unmount()

  // A failed lookup is a reachability problem with Retry, never the classic composer.
  classicRoom('Remote', mini, { log }, new Error('request timed out: groups.list'))
  await act(async () => { render(<GroupChatWorkspace group="Remote" members={mini} />) })
  await screen.findByText(CANONICAL_GROUP_LOCALES.en.groupsUnreachable)
  expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  expect(screen.queryByRole('textbox')).toBeNull()
})

it('reads the group driver again on Retry after a failed capability read', async () => {
  let reads = 0
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {
      if (++reads === 1) {throw new Error('request timed out: groups.capabilities')}

      return legacy
    }

    throw new Error(`Unexpected RPC: ${method}`)
  })
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  // A transient failure: couldn't reach the gateway, what to check, Retry; nothing is lost.
  expect(screen.getByText(CANONICAL_GROUP_LOCALES.en.groupsUnreachable)).toBeTruthy()
  expect(screen.getByRole('alert').textContent).toContain('request timed out')
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Retry' })) })
  await waitFor(() => expect(screen.getByRole('textbox')).toBeTruthy())
  expect(reads).toBe(2)
})

it('explains a refused gateway group creation in the reader\'s language with the member the gateway named, from the dialog and the gate', async () => {
  const words = 'The gateway couldn’t create this group chat. member 1 profile \'beta\': only a host-scoped connection seats a served profile without an entry'
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {return { driver: true, persistent_process: true }}

    if (method === 'groups.list') {return { rooms: [], next_offset: null }}

    if (method === 'groups.create') {
      throw Object.assign(new Error('invalid_params'), { code: 4001, data: { reason: 'invalid_params', field: 'members',
        detail: 'member 1 profile \'beta\': only a host-scoped connection seats a served profile without an entry' } })
    }

    throw new Error(`Unexpected RPC: ${method}`)
  })
  await submitDialog()
  await waitFor(() => expect(notify).toHaveBeenCalledWith({ kind: 'error', message: words }))
  cleanup()

  // And from Start new group on a classic room's gate.
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  await act(async () => { fireEvent.click(await screen.findByRole('button', { name: 'Start new group' })) })
  expect((await screen.findByRole('alert')).textContent).toBe(words)
})

// Desktop's own backend and SSH, URL and Cloud connections reach `hermes serve`
// (tui_gateway/methods_groups.py): it refuses a create with 4110, or 4123/5111 for a missing worker
// or a failure, and puts the reason in the message. The session gateway answers 4001 with data.detail.
it('keeps the gateway\'s own reason for a refused create from either server', () => {
  const labels = CANONICAL_GROUP_LOCALES.en
  const refused = labels.createRefused

  const rows: [unknown, string][] = [
    [{ code: 4001, message: 'invalid_params', data: { reason: 'invalid_params', detail: 'member 1 profile \'beta\' is parked' } },
      `${refused} member 1 profile 'beta' is parked`],
    [{ code: 4001, message: 'permission_denied', data: { reason: 'permission_denied' } }, `${refused} permission_denied`],
    // Rejections arrive as JsonRpcGatewayError, or as plain objects across the IPC bridge.
    [Object.assign(new Error('member 2 profile \'gamma\' is not served here'), { code: 4110, data: { reason: 'invalid_member' } }),
      `${refused} member 2 profile 'gamma' is not served here`],
    [Object.assign(new Error('Group Chat worker is unavailable'), { code: 4123 }), `${refused} Group Chat worker is unavailable`],
    [{ code: 5111, message: 'database is locked' }, `${refused} database is locked`],
    [new Error('request timed out: groups.create'), 'request timed out: groups.create'],
    [{ canonicalGroupProblem: 'otherConnection' }, labels.createOtherConnection]
  ]

  expect(rows.map(([error]) => canonicalGroupCreateError(error, labels))).toEqual(rows.map(([, words]) => words))
})
