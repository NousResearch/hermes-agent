import type * as HermesSdk from '@hermes/plugin-sdk'
import { host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { $canonicalGroupBindings } from './canonical-group-registry'
import { CreateGroupChatDialog } from './create-dialog'
import { $botMeta } from './data'
import { $groupChats, $groupChatWorkspace, updateGroupChat } from './group-chat'
import type * as GroupChatModule from './group-chat'
import type * as GroupChatParts from './group-chat-parts'
import { GroupChatWorkspace } from './group-chat-view'
import { translateBots } from './i18n-test-helper'

const { request, notify, openWorkspace } = vi.hoisted(() => ({ request: vi.fn(), notify: vi.fn(), openWorkspace: vi.fn() }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { en } = await import('@/i18n/en')

  return {
    ...sdk,
    host: {
      ...sdk.host, requestProfile: request, notify, openWorkspace,
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

const roster = [{ name: 'alpha', connectionId: 'local' }, { name: 'beta', connectionId: 'local' }]
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

const refused = [
  canonicalUnavailable,
  appManagedUnavailable,
  { ...legacy, authority_gateway_id: 'installation:owned' },
  { ...legacy, protocol_version: 2 },
  { ...legacy, methods: ['groups.create'] },
  { driver: false },
  { persistent_process: false },
  { driver: 0, persistent_process: false },
  { driver: 'true', persistent_process: false },
  { driver: false, persistent_process: 'false' },
  { driver: false, persistent_process: false, features: null },
  { driver: false, persistent_process: false, features: ['canonical_session_owner'] },
  null
]

beforeEach(() => {
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
    if (method === 'groups.capabilities') {return capabilities}

    if (method === 'groups.create') {return { room: { room_id: params.room_id, name: params.name, members: params.members } }}

    if (method === 'profiles.configure') {return {}}
    throw new Error(`Unexpected RPC: ${method}`)
  })
}

async function submitDialog() {
  const onCreated = vi.fn()
  const onClose = vi.fn()
  render(<CreateGroupChatDialog onClose={onClose} onCreated={onCreated} open roster={roster} />)

  for (const checkbox of screen.getAllByRole('checkbox')) {fireEvent.click(checkbox)}
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Create Group (2)' })) })

  return { onCreated, onClose }
}

it.each(refused)('keeps an unavailable or unclassified workspace out of the legacy renderer: %j', async value => {
  answer(value)
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  expect(screen.getByText(unavailable)).toBeTruthy()
  expect(screen.queryByRole('textbox')).toBeNull()
  expect((screen.getByRole('button', { name: 'Start gateway group' }) as HTMLButtonElement).disabled).toBe(true)
  expect(request.mock.calls.map(call => call[1])).toEqual(['groups.capabilities'])
})

it.each(refused)('refuses legacy creation without positive process classification: %j', async value => {
  answer(value)
  const { onCreated, onClose } = await submitDialog()
  expect(notify).toHaveBeenCalledWith({ kind: 'error', message: unavailable })
  expect(onCreated).not.toHaveBeenCalled()
  expect(onClose).not.toHaveBeenCalled()
  expect(updateGroupChat).not.toHaveBeenCalled()
  expect($botMeta.get()).toEqual({})
  expect($groupChats.get()).toEqual({})
  expect(request.mock.calls.map(call => call[1])).toEqual(['groups.capabilities'])
})

it('preserves the actual nonpersistent legacy workspace without submitting a turn', async () => {
  answer(legacy)
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  expect(screen.getByRole('textbox')).toBeTruthy()
  expect(request.mock.calls.map(call => call[1])).toEqual(['groups.capabilities'])
})

it('preserves explicit nonpersistent legacy creation', async () => {
  answer(legacy)
  const { onCreated } = await submitDialog()
  expect(onCreated).toHaveBeenCalledOnce()
  expect(updateGroupChat).toHaveBeenCalledOnce()
  expect(Object.values($groupChats.get())[0].members).toHaveLength(2)
  expect(request.mock.calls.map(call => call[1])).toEqual(['groups.capabilities', 'profiles.configure', 'profiles.configure'])
})

it('preserves canonical-ready creation without legacy metadata or room writes', async () => {
  answer({ driver: true, persistent_process: true })
  const { onCreated } = await submitDialog()
  expect(onCreated).toHaveBeenCalledOnce()
  expect(Object.values($canonicalGroupBindings.get())).toHaveLength(1)
  expect(updateGroupChat).not.toHaveBeenCalled()
  expect($botMeta.get()).toEqual({})
  expect(request.mock.calls.map(call => call[1])).toEqual(['groups.capabilities', 'groups.create'])
})

it('preserves the canonical-ready existing-room gate', async () => {
  answer({ driver: true, persistent_process: true })
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  expect((screen.getByRole('button', { name: 'Start gateway group' }) as HTMLButtonElement).disabled).toBe(false)
  expect(screen.queryByRole('textbox')).toBeNull()
})

it('shows the unavailable state after a capability read fails', async () => {
  request.mockRejectedValue(new Error('Disconnected'))
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  expect(screen.getByText(unavailable)).toBeTruthy()
  expect(screen.getByRole('alert').textContent).toContain('Disconnected')
  expect(screen.queryByRole('textbox')).toBeNull()
})

it.each(['profile', 'connection', 'gateway'] as const)('does not create a legacy room after %s changes during the read', async changed => {
  let resolve!: (value: unknown) => void
  request.mockImplementationOnce(() => new Promise(done => { resolve = done }))
  const { onCreated } = await submitDialog()
  await act(async () => {
    if (changed === 'profile') {state.profile.set('other')}

    if (changed === 'connection') {state.connectionId.set('other')}

    if (changed === 'gateway') {state.gateway.set('closed')}
    resolve(legacy)
  })
  expect(onCreated).not.toHaveBeenCalled()
  expect(updateGroupChat).not.toHaveBeenCalled()
  expect(notify).toHaveBeenCalledWith({ kind: 'error', message: unavailable })
})

it.each(['profile', 'connection', 'gateway'] as const)('retires an old workspace capability when %s changes', async changed => {
  answer(legacy)
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  expect(screen.getByRole('textbox')).toBeTruthy()
  answer(canonicalUnavailable)
  await act(async () => {
    if (changed === 'profile') {state.profile.set('other')}

    if (changed === 'connection') {state.connectionId.set('other')}

    if (changed === 'gateway') {state.gateway.set('closed')}
  })
  expect(screen.getByText(unavailable)).toBeTruthy()
  expect(screen.queryByRole('textbox')).toBeNull()
})

it('ignores a late old-profile response after the new authority has answered', async () => {
  let resolve!: (value: unknown) => void
  request.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    .mockResolvedValue(canonicalUnavailable)
  render(<GroupChatWorkspace group="Existing" members={roster} />)
  await act(async () => { state.profile.set('other') })
  await act(async () => { resolve(legacy) })
  expect(screen.getByText(unavailable)).toBeTruthy()
  expect(screen.queryByRole('textbox')).toBeNull()
})

function pendingCreation() {
  let finish!: () => void
  const serverRooms = new Map<string, unknown>()
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {return { driver: true, persistent_process: true }}

    if (method === 'groups.create') {
      const room = { room_id: params.room_id, name: params.name, members: params.members }
      serverRooms.set(room.room_id, room)

      return new Promise(resolve => { finish = () => resolve({ room }) })
    }

    throw new Error(`Unexpected RPC: ${method}`)
  })

  return { serverRooms, finish: () => finish() }
}

it('refuses a workspace click after the approved profile changes before React renders', async () => {
  answer({ driver: true, persistent_process: true })
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  const button = screen.getByRole('button', { name: 'Start gateway group' })
  await act(async () => {
    state.profile.set('other')
    fireEvent.click(button)
  })
  expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(0)
  expect($canonicalGroupBindings.get()).toEqual({})
  expect(openWorkspace).not.toHaveBeenCalled()
})

it.each(['profile', 'connection', 'gateway'] as const)('keeps a dialog-created server room without publishing its result after %s changes', async changed => {
  const pending = pendingCreation()
  const { onCreated, onClose } = await submitDialog()
  expect(pending.serverRooms.size).toBe(1)
  await act(async () => {
    if (changed === 'profile') {state.profile.set('other')}

    if (changed === 'connection') {state.connectionId.set('other')}

    if (changed === 'gateway') {state.gateway.set('closed')}
    pending.finish()
  })
  expect($canonicalGroupBindings.get()).toEqual({})
  expect(onCreated).not.toHaveBeenCalled()
  expect(onClose).not.toHaveBeenCalled()
  expect(pending.serverRooms.size).toBe(1)
  expect(request.mock.calls.map(call => call[1])).toEqual(['groups.capabilities', 'groups.create'])
  expect(request.mock.calls[1][0]).toMatchObject({ connectionId: 'local', profile: 'default', targetProfile: 'default' })
})

it.each(['profile', 'connection', 'gateway'] as const)('keeps a workspace-created server room without opening it after %s changes', async changed => {
  const pending = pendingCreation()
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Start gateway group' })) })
  expect(pending.serverRooms.size).toBe(1)
  await act(async () => {
    if (changed === 'profile') {state.profile.set('other')}

    if (changed === 'connection') {state.connectionId.set('other')}

    if (changed === 'gateway') {state.gateway.set('closed')}
    $groupChatWorkspace.set('Other foreground room')
    pending.finish()
  })
  expect($canonicalGroupBindings.get()).toEqual({})
  expect(openWorkspace).not.toHaveBeenCalled()
  expect($groupChatWorkspace.get()).toBe('Other foreground room')
  expect(pending.serverRooms.size).toBe(1)
  const creates = request.mock.calls.filter(call => call[1] === 'groups.create')
  expect(creates).toHaveLength(1)
  expect(creates[0][0]).toMatchObject({ connectionId: 'local', profile: 'default', targetProfile: 'default' })
  expect(request.mock.calls.every(call => ['groups.capabilities', 'groups.create'].includes(call[1]))).toBe(true)
})

it.each(['dialog', 'workspace'] as const)('publishes the single delayed %s result when its approved source stays current', async caller => {
  const pending = pendingCreation()
  let dialog: Awaited<ReturnType<typeof submitDialog>> | undefined

  if (caller === 'dialog') {
    dialog = await submitDialog()
  } else {
    await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Start gateway group' })) })
  }

  await act(async () => { pending.finish() })
  expect(Object.values($canonicalGroupBindings.get())).toHaveLength(1)
  expect(pending.serverRooms.size).toBe(1)
  expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(1)

  if (dialog) {
    expect(dialog.onCreated).toHaveBeenCalledOnce()
    expect(dialog.onClose).toHaveBeenCalledOnce()
  } else {
    expect(openWorkspace).toHaveBeenCalledOnce()
  }
})
