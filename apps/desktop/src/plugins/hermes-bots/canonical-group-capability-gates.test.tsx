import type * as HermesSdk from '@hermes/plugin-sdk'
import { host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { $canonicalGroupBindings, $canonicalGroupNames, CanonicalGroupList, forgetCanonicalGroup, registerCanonicalGroup } from './canonical-group-registry'
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
  $canonicalGroupNames.set({})
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
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup })) })

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
  expect((screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.startGatewayGroup }) as HTMLButtonElement).disabled).toBe(true)
  cleanup()

  const { onCreated, onClose } = await submitDialog()
  expect(screen.getByRole('alert').textContent).toContain(unavailable)
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
    // Each discovery mount reads the current socket's surface; classic
    // workspace gates still share the latest capability classification.
    expect(request.mock.calls.filter(call => call[1].startsWith('groups.'))).toHaveLength(2)
    await act(async () => { fireEvent.click(screen.getAllByRole('button', { name: CANONICAL_GROUP_LOCALES.en.refreshGroups })[0]) })
    expect(request.mock.calls.filter(call => call[1].startsWith('groups.'))).toHaveLength(3)
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
  expect(screen.queryByRole('button', { name: CANONICAL_GROUP_LOCALES.en.startGatewayGroup })).toBeNull()
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
    expect(screen.getByText(CANONICAL_GROUP_LOCALES.en.classicConnection)).toBeTruthy()
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'info', message: CANONICAL_GROUP_LOCALES.en.classicConnection }))
    expect(Object.values($groupChats.get())[0].members?.map(member => member.connectionId)).toEqual(['local', 'remote'])
  }
})

it('shows a generic create refusal without guessing a profile setup failure or creating classic state', async () => {
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}
    throw Object.assign(new Error('invalid_params'), { code: 4001, data: { reason: 'invalid_params' } })
  })
  const { onCreated, onClose } = await submitDialog()
  expect(screen.getByRole('alert').textContent).toContain(CANONICAL_GROUP_LOCALES.en.createRefused)
  expect(screen.queryByRole('link')).toBeNull()
  expect(onCreated).not.toHaveBeenCalled()
  expect(onClose).not.toHaveBeenCalled()
  expect(updateGroupChat).not.toHaveBeenCalled()
  expect($groupChats.get()).toEqual({})
  expect($canonicalGroupBindings.get()).toEqual({})
  cleanup()
  await act(async () => { render(<GroupChatWorkspace group="Existing" members={roster} />) })
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.startGatewayGroup })) })
  expect(screen.getByRole('alert').textContent).toContain(CANONICAL_GROUP_LOCALES.en.createRefused)
  expect(screen.queryByRole('link')).toBeNull()
  expect(screen.getByRole('textbox')).toBeTruthy()
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
  const button = screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.startGatewayGroup })
  expect((button as HTMLButtonElement).disabled).toBe(false)
  // Click after the source moved but before React re-renders: the stale capability must not create.
  await act(async () => { moveSource(kind); fireEvent.click(button) })
  expect(request.mock.calls.filter(call => call[1] === 'groups.create')).toHaveLength(0)
  expect(pending.serverRooms.size).toBe(0)
  expect(openWorkspace).not.toHaveBeenCalled()
})


it('keeps friendly Bot identities on the created group and never navigates after closing a pending creation', async () => {
  const pending = pendingCreation()
  const onCreated = vi.fn()
  const onClose = vi.fn()
  render(<CreateGroupChatDialog onClose={onClose} onCreated={onCreated} open roster={[
    { ...roster[0], display_name: 'Mira Bot' }, { ...roster[1], display_name: 'Atlas Bot' }
  ]} />)

  for (const checkbox of screen.getAllByRole('checkbox')) {fireEvent.click(checkbox)}
  fireEvent.change(screen.getByRole('textbox', { name: CANONICAL_GROUP_LOCALES.en.nameOptional }), { target: { value: 'Autumn launch' } })
  await act(async () => {fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup }))})
  expect((screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.creatingGroup }) as HTMLButtonElement).disabled).toBe(true)
  const createCall = request.mock.calls.find(call => call[1] === 'groups.create')
  expect(createCall?.[2].members.map((member: { display_name: string }) => member.display_name)).toEqual(['Mira Bot', 'Atlas Bot'])
  fireEvent.click(screen.getAllByRole('button', { name: CANONICAL_GROUP_LOCALES.en.close })[0])
  await act(async () => {pending.finish()})
  expect(pending.serverRooms.size).toBe(1)
  expect(onClose).toHaveBeenCalledOnce()
  expect(onCreated).not.toHaveBeenCalled()
  expect(Object.values($canonicalGroupBindings.get())).toHaveLength(0)
})

it('releases creation when its connection disappears immediately before the click', async () => {
  answer(CANONICAL_GROUP_CAPABILITIES)
  const onCreated = vi.fn()
  render(<CreateGroupChatDialog onClose={vi.fn()} onCreated={onCreated} open roster={roster} />)

  for (const checkbox of screen.getAllByRole('checkbox')) {fireEvent.click(checkbox)}
  const create = screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup })
  await act(async () => {state.connectionId.set(null); fireEvent.click(create)})
  expect(request.mock.calls.some(call => call[1] === 'groups.create')).toBe(false)
  expect(screen.queryByRole('button', { name: CANONICAL_GROUP_LOCALES.en.creatingGroup })).toBeNull()
  expect(onCreated).not.toHaveBeenCalled()
})

it('clears a recovered setup error without losing the selected Bots or group name', async () => {
  const desktop = window.hermesDesktop
  const recover = vi.fn().mockResolvedValueOnce({ ok: true }).mockResolvedValueOnce({ ok: false, pending: true, reason: 'setup_journal_unreadable' }).mockResolvedValue({ ok: true })
  const create = vi.fn().mockResolvedValue({ ok: false, reason: 'setup_journal_write_failed' })
  window.hermesDesktop = { roomSetup: { recover, create } } as unknown as typeof window.hermesDesktop
  answer(CANONICAL_GROUP_CAPABILITIES)

  try {
    await act(async () => {render(<CreateGroupChatDialog onClose={vi.fn()} open roster={[
      { name: 'default', handle: 'atlas', connectionId: 'local', display_name: 'Atlas Bot' },
      { name: 'default', handle: 'mira', connectionId: 'remote', display_name: 'Mira Bot' }
    ]} />)})

    for (const checkbox of screen.getAllByRole('checkbox')) {fireEvent.click(checkbox)}
    fireEvent.change(screen.getByRole('textbox', { name: CANONICAL_GROUP_LOCALES.en.nameOptional }), { target: { value: 'Harbor launch' } })
    await act(async () => {fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup }))})
    expect(create).toHaveBeenCalledOnce()
    expect(screen.getByRole('alert').textContent).toContain(CANONICAL_GROUP_LOCALES.en.peerSetupStorage)
    await act(async () => {fireEvent.click(screen.getByRole('button', { name: 'Retry' }))})
    expect(screen.queryByRole('alert')).toBeNull()
    expect((screen.getByRole('textbox', { name: CANONICAL_GROUP_LOCALES.en.nameOptional }) as HTMLInputElement).value).toBe('Harbor launch')
    expect(screen.getAllByRole('checkbox').every(box => box.getAttribute('aria-checked') === 'true')).toBe(true)
    expect((screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup }) as HTMLButtonElement).disabled).toBe(false)
  } finally {window.hermesDesktop = desktop}
})


it('shows a newly created group in the mounted sidebar and follows its rename without discovering a foreign owner', async () => {
  request.mockImplementation(async (_route, method) => method === 'groups.capabilities' ? CANONICAL_GROUP_CAPABILITIES : { rooms: [], next_offset: null })
  const onOpen = vi.fn()
  await act(async () => {render(<CanonicalGroupList onOpen={onOpen} />)})
  const room = { room_id: 'fresh-group', name: 'Harbor launch', members: [] }
  const route = { connectionId: 'local', profile: 'default' }
  let key = ''
  act(() => {key = registerCanonicalGroup(route, room)})
  fireEvent.click(screen.getByRole('button', { name: room.name }))
  expect(onOpen).toHaveBeenCalledWith(key)
  act(() => {registerCanonicalGroup(route, { ...room, name: 'Friday launch' })})
  expect(screen.queryByRole('button', { name: room.name })).toBeNull()
  expect(screen.getByRole('button', { name: 'Friday launch' })).toBeTruthy()
  act(() => {registerCanonicalGroup({ connectionId: 'another-owner', profile: 'default' }, { ...room, name: 'Private group' })})
  expect(screen.queryByRole('button', { name: 'Private group' })).toBeNull()
  act(() => {forgetCanonicalGroup({ ...route, roomId: room.room_id })})
  expect(screen.queryByRole('button', { name: 'Friday launch' })).toBeNull()
})

it('reconciles a late discovery with creation, rename and retirement that happened while it was pending', async () => {
  const route = { connectionId: 'local', profile: 'default' }
  const renamed = { room_id: 'renamed', name: 'Old name', members: [] }
  const ended = { room_id: 'ended', name: 'Ended group', members: [] }
  const transient = { room_id: 'transient', name: 'Created then ended', members: [] }
  registerCanonicalGroup(route, renamed)
  const endedKey = registerCanonicalGroup(route, ended)
  let resolveList!: (value: unknown) => void
  request.mockImplementation(async (_route, method) => method === 'groups.capabilities' ? CANONICAL_GROUP_CAPABILITIES
    : new Promise(resolve => {resolveList = resolve}))
  await act(async () => {render(<CanonicalGroupList onOpen={vi.fn()} />)})
  let transientKey = ''
  act(() => {
    registerCanonicalGroup(route, { room_id: 'fresh', name: 'New group', members: [] })
    registerCanonicalGroup(route, { ...renamed, name: 'Current name' })
    transientKey = registerCanonicalGroup(route, transient)
    forgetCanonicalGroup({ ...route, roomId: transient.room_id })
    forgetCanonicalGroup({ ...route, roomId: ended.room_id })
  })
  await act(async () => {resolveList({ rooms: [renamed, ended, transient], next_offset: null })})
  expect(screen.getByRole('button', { name: 'New group' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Current name' })).toBeTruthy()
  expect(screen.queryByRole('alert')).toBeNull()
  expect(screen.queryByRole('button', { name: 'Old name' })).toBeNull()
  expect(screen.queryByRole('button', { name: ended.name })).toBeNull()
  expect(screen.queryByRole('button', { name: transient.name })).toBeNull()
  expect($canonicalGroupBindings.get()[endedKey]).toBeUndefined()
  expect($canonicalGroupBindings.get()[transientKey]).toBeUndefined()
})

it('keeps the existing classic transcript and composer when explicitly starting a separate gateway group', async () => {
  answer(CANONICAL_GROUP_CAPABILITIES)
  const log = [
    { id: 'classic-message', from: { kind: 'user' as const, name: 'You' }, text: 'Earlier classic conversation', at: 1 }
  ]
  $groupChats.set({ Existing: { log, watermarks: {}, sessions: {} } })
  await act(async () => {
    render(<GroupChatWorkspace group="Existing" members={roster} />)
  })
  expect(screen.getByText('Earlier classic conversation')).toBeTruthy()
  const composer = screen.getByRole('textbox') as HTMLTextAreaElement
  fireEvent.change(composer, { target: { value: 'Continue classic draft' } })
  expect(composer.disabled).toBe(false)
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.startGatewayGroup }))
  })
  expect($groupChats.get().Existing.log).toEqual(log)
  expect(composer.value).toBe('Continue classic draft')
  const create = request.mock.calls.find(call => call[1] === 'groups.create')!
  expect(create[2]).not.toHaveProperty('history')
  expect(create[2]).not.toHaveProperty('messages')
  expect(Object.keys($canonicalGroupBindings.get())[0]).not.toBe('Existing')
  expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(false)
})

it('lets your other computers continue a new group in one step, on by default, only when the host can designate them', async () => {
  const desktop = window.hermesDesktop
  const room = { room_id: 'harbor', name: 'Harbor launch', members: [] }
  const create = vi.fn().mockResolvedValue({ ok: true, room, successors: 'failed' })
  window.hermesDesktop = { roomSetup: { recover: vi.fn().mockResolvedValue({ ok: true }), create } } as unknown as typeof window.hermesDesktop
  vi.mocked(host.connections).mockResolvedValue([{ id: 'local', label: 'Mac mini', installId: 'a'.repeat(32) }] as never)

  const peers = [{ name: 'default', handle: 'atlas', connectionId: 'local', display_name: 'Atlas Bot' },
    { name: 'default', handle: 'mira', connectionId: 'laptop', display_name: 'Mira Bot' }]

  const layer7 = { ...CANONICAL_GROUP_CAPABILITIES, methods: [...CANONICAL_GROUP_CAPABILITIES.methods, 'groups.succession.status', 'groups.custody.designate'] }

  try {
    answer(CANONICAL_GROUP_CAPABILITIES)
    await act(async () => {render(<CreateGroupChatDialog onClose={vi.fn()} open roster={peers} />)})

    for (const checkbox of screen.getAllByRole('checkbox')) {fireEvent.click(checkbox)}
    await act(async () => {await Promise.resolve()})
    expect(screen.queryByRole('switch', { name: 'Let my computers continue this group' })).toBeNull()
    cleanup()

    activation.epoch++
    answer(layer7)
    await act(async () => {render(<CreateGroupChatDialog onClose={vi.fn()} open roster={peers} />)})
    expect(screen.queryByRole('switch')).toBeNull()

    for (const checkbox of screen.getAllByRole('checkbox')) {fireEvent.click(checkbox)}
    const toggle = await screen.findByRole('switch', { name: 'Let my computers continue this group' })
    expect(toggle.getAttribute('aria-checked')).toBe('true')
    expect(screen.getByText('Your other computers keep a full copy of this group and can continue it if Mac mini goes offline.')).toBeTruthy()
    await act(async () => {fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup }))})
    expect(create.mock.calls[0][0]).toMatchObject({ successor: true })
    expect(notify).toHaveBeenCalledWith({ kind: 'info', message: '“Harbor launch” is ready, but your other computers can’t continue it yet. You can turn this on in Backup copies.' })
    cleanup()

    create.mockResolvedValue({ ok: true, room, successors: undefined })
    notify.mockReset()
    await act(async () => {render(<CreateGroupChatDialog onClose={vi.fn()} open roster={peers} />)})

    for (const checkbox of screen.getAllByRole('checkbox')) {fireEvent.click(checkbox)}
    fireEvent.click(await screen.findByRole('switch', { name: 'Let my computers continue this group' }))
    await act(async () => {fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup }))})
    expect(create.mock.calls[1][0]).not.toHaveProperty('successor')
    expect(notify).not.toHaveBeenCalled()
  } finally {window.hermesDesktop = desktop}
})

/** A gateway that can host groups and join one hosted elsewhere, run by `operator`. */
const computerCapabilities = (install: string, operator: string, extra: Record<string, unknown> = {}) => ({
  ...CANONICAL_GROUP_CAPABILITIES, authority_gateway_id: `install:${install.repeat(32)}`, features: ['peer_setup_recovery'],
  room_link: { enabled: true, authentication: 'proof-v2', endpoint: { available: true, url: `https://${install}.example` },
    catalog: { persistent_process: true, installation_id: `install:${install.repeat(32)}`, text: true, attachments: false, catalog_digest: install } },
  room_identity: { operator_name: operator, always_on: false }, ...extra })

function threeComputers(byConnection: Record<string, unknown>) {
  const room = { room_id: 'harbor', name: 'Harbor launch', members: [] }
  const create = vi.fn().mockResolvedValue({ ok: true, room })
  window.hermesDesktop = { roomSetup: { recover: vi.fn().mockResolvedValue({ ok: true }), create } } as unknown as typeof window.hermesDesktop
  vi.mocked(host.connections).mockResolvedValue([{ id: 'local', label: 'Mac mini', installId: 'a'.repeat(32) },
    { id: 'vps', label: 'Home VPS', installId: 'b'.repeat(32) }, { id: 'laptop', label: 'Laptop', installId: 'c'.repeat(32) }] as never)
  request.mockImplementation(async (route: { connectionId: string }, method: string) => {
    if (method === 'groups.capabilities') {return byConnection[route.connectionId]}
    throw new Error(`Unexpected RPC: ${method}`)
  })

  return create
}

const acrossComputers = [{ name: 'default', handle: 'atlas', connectionId: 'local', display_name: 'Atlas Bot' },
  { name: 'default', handle: 'mira', connectionId: 'vps', display_name: 'Mira Bot' },
  { name: 'default', handle: 'rex', connectionId: 'laptop', display_name: 'Rex Bot' }]

async function chooseAll() {
  for (const checkbox of screen.getAllByRole('checkbox')) {fireEvent.click(checkbox)}
  await act(async () => {await new Promise(resolve => setTimeout(resolve, 0))})
}

it('hosts a group across computers on your always-on computer by default, and creates it there', async () => {
  const desktop = window.hermesDesktop

  const create = threeComputers({ local: computerCapabilities('a', 'Dana'), laptop: computerCapabilities('c', 'Dana'),
    vps: computerCapabilities('b', 'Dana', { room_identity: { operator_name: 'Dana', always_on: true } }) })

  try {
    await act(async () => {render(<CreateGroupChatDialog onClose={vi.fn()} open roster={acrossComputers} />)})
    await chooseAll()
    const vps = await screen.findByRole('radio', { name: 'Home VPS' }) as HTMLInputElement
    expect(vps.checked).toBe(true)
    expect(screen.getByText('Hosted on Home VPS because it’s always on. The group keeps running when this computer sleeps.')).toBeTruthy()
    await act(async () => {fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup }))})
    expect(create.mock.calls[0][0].home).toEqual({ connectionId: 'vps', profile: 'default' })
    // Listed where you created it, bound to the computer that hosts it.
    expect(Object.entries($canonicalGroupBindings.get())).toEqual([[expect.stringContaining('local'), { connectionId: 'vps', profile: 'default', roomId: 'harbor' }]])
  } finally {window.hermesDesktop = desktop}
})

it('keeps a computer that can’t host visible but disabled, and says what a sleeping host means', async () => {
  const desktop = window.hermesDesktop

  const create = threeComputers({ local: computerCapabilities('a', 'Dana'), laptop: computerCapabilities('c', 'Dana', { driver: false }),
    vps: computerCapabilities('b', 'Dana', { room_identity: { operator_name: 'Dana', always_on: true } }) })

  try {
    await act(async () => {render(<CreateGroupChatDialog onClose={vi.fn()} open roster={acrossComputers} />)})
    await chooseAll()
    const laptop = await screen.findByRole('radio', { name: /^Laptop/ }) as HTMLInputElement
    expect(laptop.disabled).toBe(true)
    expect(screen.getByText('Laptop can’t host this group.')).toBeTruthy()

    fireEvent.click(screen.getByRole('radio', { name: 'Mac mini' }))
    expect(screen.getByText('Mac mini sleeps. While it’s asleep the group moves to Home VPS, and Mac mini’s Bots wait until it moves back.')).toBeTruthy()
    await act(async () => {fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup }))})
    expect(create.mock.calls[0][0].home).toEqual({ connectionId: 'local', profile: 'default' })
    expect(Object.values($canonicalGroupBindings.get())).toEqual([{ connectionId: 'local', profile: 'default', roomId: 'harbor' }])
  } finally {window.hermesDesktop = desktop}
})

it('suggests an always-on computer whose owner isn’t known instead of choosing it, and names the computer, not a person', async () => {
  const desktop = window.hermesDesktop
  // This computer is Dana's; the others report no operator name, so whose they are isn't known.
  const unnamed = (install: string, extra: Record<string, unknown> = {}) => computerCapabilities(install, '', extra)

  const create = threeComputers({ local: computerCapabilities('a', 'Dana'), laptop: unnamed('c'),
    vps: unnamed('b', { room_identity: { operator_name: null, always_on: true } }) })

  try {
    await act(async () => {render(<CreateGroupChatDialog onClose={vi.fn()} open roster={acrossComputers} />)})
    await chooseAll()
    expect((await screen.findByRole('radio', { name: 'Mac mini' }) as HTMLInputElement).checked).toBe(true)
    expect(screen.queryByText(/because it’s always on/)).toBeNull()
    expect(screen.getByText('Home VPS will keep a full copy of this group’s history, including earlier messages.')).toBeTruthy()
    expect(screen.getByText('Laptop will keep a full copy of this group’s history, including earlier messages.')).toBeTruthy()
    expect(screen.queryByText(/’s computer will keep/)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Tip: host on Home VPS, which is always on, so the group keeps running when this computer sleeps.' }))
    expect((screen.getByRole('radio', { name: 'Home VPS' }) as HTMLInputElement).checked).toBe(true)
    expect(screen.getByText('Hosted on Home VPS because it’s always on. The group keeps running when this computer sleeps.')).toBeTruthy()
    expect(screen.queryByRole('button', { name: /^Tip:/ })).toBeNull()
    await act(async () => {fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup }))})
    expect(create.mock.calls[0][0].home).toEqual({ connectionId: 'vps', profile: 'default' })
  } finally {window.hermesDesktop = desktop}
})

it('never preselects another person’s computer, names it before adding it, and offers no choice when nothing else could host', async () => {
  const desktop = window.hermesDesktop
  // Mira's always-on computer is Sam's, and can't join groups hosted elsewhere.
  const sams = computerCapabilities('b', 'Sam', { room_link: undefined, room_identity: { operator_name: 'Sam', always_on: true } })
  let create = threeComputers({ local: computerCapabilities('a', 'Dana'), laptop: computerCapabilities('c', 'Dana'), vps: sams })

  try {
    await act(async () => {render(<CreateGroupChatDialog onClose={vi.fn()} open roster={acrossComputers} />)})
    await chooseAll()
    expect(await screen.findByText('Sam’s computer will keep a full copy of this group’s history, including earlier messages.')).toBeTruthy()
    expect((await screen.findByRole('radio', { name: 'Mac mini' }) as HTMLInputElement).checked).toBe(true)
    expect((screen.getByRole('radio', { name: 'Home VPS' }) as HTMLInputElement).checked).toBe(false)
    expect(screen.getByText('Home VPS isn’t set up to join a group on another computer yet.')).toBeTruthy()
    expect(screen.queryByText(/because it’s always on/)).toBeNull()
    cleanup()

    // Neither other computer can join a group elsewhere: no choice, and the group is created as before.
    activation.epoch++
    create = threeComputers({ local: computerCapabilities('a', 'Dana'), laptop: { ...computerCapabilities('c', 'Dana'), room_link: undefined }, vps: sams })
    await act(async () => {render(<CreateGroupChatDialog onClose={vi.fn()} open roster={acrossComputers} />)})
    await chooseAll()
    await screen.findByText('Sam’s computer will keep a full copy of this group’s history, including earlier messages.')
    expect(screen.queryByRole('radio')).toBeNull()
    await act(async () => {fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup }))})
    expect(create.mock.calls[0][0].home).toEqual({ connectionId: 'local', profile: 'default' })
  } finally {window.hermesDesktop = desktop}
})
