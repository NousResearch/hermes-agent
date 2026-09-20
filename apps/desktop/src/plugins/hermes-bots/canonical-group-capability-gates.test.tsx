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

const { request, notify, openWorkspace, activation } = vi.hoisted(() => ({ request: vi.fn(), notify: vi.fn(), openWorkspace: vi.fn(), activation: { epoch: 1 } }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { en } = await import('@/i18n/en')

  return {
    ...sdk,
    gatewayActivationEpoch: () => activation.epoch,
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

const refused = [canonicalUnavailable, appManagedUnavailable, { ...legacy, methods: ['groups.create'] }, { driver: 'true', persistent_process: false }, null]

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

it.each(refused)('classifies %j as unavailable on both surfaces: no legacy renderer, no legacy creation', async value => {
  answer(value)
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  expect(screen.getByText(unavailable)).toBeTruthy()
  expect(screen.queryByRole('textbox')).toBeNull()
  expect((screen.getByRole('button', { name: 'Start gateway group' }) as HTMLButtonElement).disabled).toBe(true)
  cleanup()

  const { onCreated, onClose } = await submitDialog()
  expect(notify).toHaveBeenCalledWith({ kind: 'error', message: unavailable })
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
  const button = screen.getByRole('button', { name: 'Start gateway group' })
  expect((button as HTMLButtonElement).disabled).toBe(false)
  // Click after the source moved but before React re-renders: the stale capability must not create.
  await act(async () => { moveSource(kind); fireEvent.click(button) })
  expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(0)
  expect(pending.serverRooms.size).toBe(0)
  expect(openWorkspace).not.toHaveBeenCalled()
})
