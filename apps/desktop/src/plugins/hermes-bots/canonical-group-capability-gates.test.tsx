import type * as HermesSdk from '@hermes/plugin-sdk'
import { host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { $canonicalGroupBindings, CanonicalGroupList } from './canonical-group-registry'
import { CreateGroupChatDialog } from './create-dialog'
import { $botMeta } from './data'
import { $groupChats, $groupChatWorkspace, updateGroupChat } from './group-chat'
import type * as GroupChatModule from './group-chat'
import type * as GroupChatParts from './group-chat-parts'
import { GroupChatWorkspace } from './group-chat-view'
import { CANONICAL_GROUP_CAPABILITIES, STANDALONE_GROUP_CAPABILITIES } from './group-test-utils'
import { translateBots } from './i18n-test-helper'

const { request, notify, openWorkspace, activation } = vi.hoisted(() => ({ request: vi.fn(), notify: vi.fn(), openWorkspace: vi.fn(), activation: { epoch: 1 } }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { en } = await import('@/i18n/en')
  const { captureGroupRequests } = await import('./group-test-utils')
  const captured = captureGroupRequests(request)

  return {
    ...sdk,
    gatewayActivationEpoch: () => activation.epoch,
    host: {
      ...sdk.host, requestProfile: captured.request, notify, openWorkspace,
      request: (method: string, params?: Record<string, unknown>) => captured.request(null, method, params),
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

const canonicalUnavailable = { ...CANONICAL_GROUP_CAPABILITIES, driver: false }
const appManagedUnavailable = { ...canonicalUnavailable, persistent_process: false }
const legacy = STANDALONE_GROUP_CAPABILITIES
const refused = [canonicalUnavailable, appManagedUnavailable, { ...CANONICAL_GROUP_CAPABILITIES, driver: undefined }, { ...CANONICAL_GROUP_CAPABILITIES, driver: 'true' }, null]

beforeEach(() => {
  activation.epoch++
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

async function submitDialog(members = roster) {
  const onCreated = vi.fn()
  const onClose = vi.fn()
  render(<CreateGroupChatDialog onClose={onClose} onCreated={onCreated} open roster={members} />)

  for (const checkbox of screen.getAllByRole('checkbox')) {fireEvent.click(checkbox)}
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Create Group (2)' })) })

  return { onCreated, onClose }
}

function pendingCreation() {
  let finish!: () => void
  const serverRooms = new Map<string, unknown>()
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}

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

  answer(CANONICAL_GROUP_CAPABILITIES)
  const { onCreated } = await submitDialog()
  expect(onCreated).toHaveBeenCalledOnce()
  expect(Object.values($canonicalGroupBindings.get())).toHaveLength(1)
  expect(updateGroupChat).toHaveBeenCalledOnce()
  expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(1)
})

it('keeps a standalone classic composer through transient failures without retargeting its connection or profile', async () => {
  vi.useFakeTimers()

  try {
    answer(STANDALONE_GROUP_CAPABILITIES)
    await act(async () => { render(<CanonicalGroupList onOpen={vi.fn()} />) })
    await act(async () => { render(<GroupChatWorkspace group="Standalone" members={roster} />) })
    const composer = screen.getByRole('textbox')
    fireEvent.change(composer, { target: { value: 'Keep this draft' } })
    await act(async () => { render(<GroupChatWorkspace group="Second standalone" members={roster} />) })
    await act(async () => { render(<CanonicalGroupList onOpen={vi.fn()} />) })
    expect(request.mock.calls.filter(call => call[1].startsWith('groups.'))).toHaveLength(1)
    await act(async () => { fireEvent.click(screen.getAllByRole('button', { name: CANONICAL_GROUP_LOCALES.en.refreshGroups })[0]) })
    expect(request.mock.calls.filter(call => call[1].startsWith('groups.'))).toHaveLength(2)
    request.mockImplementation(async () => { throw new Error('timeout') })
    await act(async () => { state.gateway.set('closed') })
    expect(screen.getAllByRole('textbox')).toContain(composer)
    await act(async () => { activation.epoch++; state.gateway.set('open') })
    expect((composer as HTMLTextAreaElement).value).toBe('Keep this draft')
    expect(screen.getAllByRole('textbox')).toContain(composer)
    expect(request.mock.calls.every(call => call[1] === 'groups.capabilities')).toBe(true)
    expect(request.mock.calls.every(call => call[0]?.connectionId === 'local' && call[2]?.profile === 'default')).toBe(true)
    cleanup()
    await act(async () => { state.profile.set('fresh-profile'); render(<GroupChatWorkspace group="Fresh" members={roster} />) })
    expect(screen.queryByRole('textbox')).toBeNull()
    expect(screen.getByRole('alert').textContent).toContain('timeout')
    answer(STANDALONE_GROUP_CAPABILITIES)
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Retry now' })) })
    expect(screen.getByRole('textbox')).toBeTruthy()
    expect((await submitDialog()).onCreated).toHaveBeenCalledOnce()
    expect(request.mock.calls.every(call => !call[1].startsWith('groups.') || call[1] === 'groups.capabilities')).toBe(true)
  } finally { vi.useRealTimers() }
})

it('keeps a mixed-connection classic composer on a canonical surface without offering creation', async () => {
  answer(CANONICAL_GROUP_CAPABILITIES)
  const mixed = [roster[0], { ...roster[1], connectionId: 'remote' }]
  await act(async () => { render(<GroupChatWorkspace group="Across machines" members={mixed} />) })
  const composer = screen.getByRole('textbox')
  fireEvent.change(composer, { target: { value: 'Keep working across machines' } })
  expect((composer as HTMLTextAreaElement).value).toBe('Keep working across machines')
  expect(screen.queryByRole('button', { name: 'Start gateway group' })).toBeNull()
  expect(request.mock.calls.map(call => call[1])).toEqual(['groups.capabilities'])
})

it.each([false, true])('selects the canonical or classic creation path for a mixed roster: %s', async mixed => {
  answer(CANONICAL_GROUP_CAPABILITIES)
  const members = [roster[0], { ...roster[1], connectionId: mixed ? 'remote' : 'local' }]
  const { onCreated, onClose } = await submitDialog(members)
  expect(onCreated).toHaveBeenCalledOnce()
  expect(onClose).toHaveBeenCalledOnce()
  expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(mixed ? 0 : 1)
  expect(updateGroupChat).toHaveBeenCalledTimes(mixed ? 1 : 0)
  expect(Object.values($canonicalGroupBindings.get())).toHaveLength(mixed ? 0 : 1)

  if (mixed) {
    expect(screen.getByRole('status').textContent).toContain('another connection')
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'info', message: expect.stringContaining('another connection') }))
    expect(Object.values($groupChats.get())[0].members?.map(member => member.connectionId)).toEqual(['local', 'remote'])
  }
})

it('explains a hosted-profile create refusal in the dialog and room gate without creating classic state', async () => {
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}
    throw Object.assign(new Error('invalid_params'), { code: 4001, data: { reason: 'invalid_params' } })
  })
  const { onCreated, onClose } = await submitDialog()
  expect(screen.getByRole('alert').textContent).toContain('hosted_rooms.profiles')
  expect(screen.getByRole('link').getAttribute('href')).toBe('https://hermes-agent.nousresearch.com/docs/developer-guide/hosted-profile-owners')
  expect(onCreated).not.toHaveBeenCalled()
  expect(onClose).not.toHaveBeenCalled()
  expect(updateGroupChat).not.toHaveBeenCalled()
  expect($groupChats.get()).toEqual({})
  expect($canonicalGroupBindings.get()).toEqual({})
  cleanup()
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Start gateway group' })) })
  expect(screen.getByRole('alert').textContent).toContain('hosted_rooms.profiles')
  expect(screen.getByRole('link').getAttribute('href')).toBe('https://hermes-agent.nousresearch.com/docs/developer-guide/hosted-profile-owners')
  expect(screen.queryByRole('textbox')).toBeNull()
  expect(updateGroupChat).not.toHaveBeenCalled()
  expect(openWorkspace).not.toHaveBeenCalled()
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
