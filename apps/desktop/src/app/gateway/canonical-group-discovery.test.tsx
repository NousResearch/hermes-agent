import { gatewayActivationEpoch, host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $canonicalGroupBindings, $canonicalGroupNames, CanonicalGroupList, forgetCanonicalGroup } from '@/plugins/hermes-bots/canonical-group-registry'
import { readGroupExecutionMode } from '@/plugins/hermes-bots/canonical-groups'
import { GroupChatWorkspace } from '@/plugins/hermes-bots/group-chat-view'
import { CANONICAL_GROUP_CAPABILITIES, STANDALONE_GROUP_CAPABILITIES } from '@/plugins/hermes-bots/group-test-utils'
import type * as BotsI18n from '@/plugins/hermes-bots/i18n'
import {
  configureGatewayRegistry,
  ensureGatewayForProfile,
  reportPrimaryGatewayState,
  setPrimaryGateway,
  setPrimaryGatewayConnection
} from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { setConnection } from '@/store/session'

vi.mock('@/plugins/hermes-bots/canonical-group-labels', () => ({
  useCanonicalGroupLabels: () => ({ refreshGroups: 'Refresh gateway groups' })
}))
vi.mock('@/plugins/hermes-bots/i18n', async importOriginal => {
  const actual = await importOriginal<typeof BotsI18n>()
  const english = actual.BOTS_LOCALES.en as unknown as ReturnType<typeof actual.useBots>

  return { ...actual, useBots: () => english }
})

const request = vi.fn()
let connectionSequence = 0
let connectionId: string

const gateway = {
  connectionState: 'connecting',
  request: async (method: string, params: Record<string, unknown>) => {
    if (gateway.connectionState !== 'open') {throw new Error('Hermes gateway unavailable')}

    return request(method, params)
  }
}

function changeSocket(state: 'connecting' | 'open' | 'closed') {
  gateway.connectionState = state
  reportPrimaryGatewayState(state)
}

beforeEach(() => {
  connectionId = `discovery-${++connectionSequence}`
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway(gateway as never, 'default')
  setPrimaryGatewayConnection({ connectionId, mode: 'local' })
  setConnection({ connectionId, mode: 'local', profile: 'default', port: 12345 } as never)
  $activeGatewayProfile.set('default')
  changeSocket('connecting')
  $canonicalGroupBindings.set({})
  $canonicalGroupNames.set({})
  request.mockReset()
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

afterEach(() => {
  cleanup()
  setPrimaryGateway(null)
  setConnection(null)
  reportPrimaryGatewayState('closed')
  vi.restoreAllMocks()
})

it('discovers retained rooms when the actual primary socket opens after the list mounts', async () => {
  const route = { connectionId, profile: 'default' }
  // An earlier surface can have classified the not-yet-open primary as
  // unavailable. Socket readiness does not advance the activation epoch.
  const epoch = gatewayActivationEpoch()
  expect((await readGroupExecutionMode(route, epoch)).mode).toBe('unavailable')
  expect(request).not.toHaveBeenCalled()
  request.mockImplementation(async method => method === 'groups.capabilities'
    ? CANONICAL_GROUP_CAPABILITIES
    : { rooms: [{ room_id: 'retained', name: 'Retained room', members: [] }], next_offset: null })

  render(<CanonicalGroupList onOpen={vi.fn()} />)
  await act(async () => {})
  expect(host.state.gateway.get()).toBe('connecting')
  expect(request).not.toHaveBeenCalled()

  await act(async () => { changeSocket('open') })
  await screen.findByRole('button', { name: 'Retained room' })
  expect(gatewayActivationEpoch()).toBe(epoch)
  expect(request.mock.calls).toEqual([
    ['groups.capabilities', { profile: 'default' }],
    ['groups.list', { limit: 100, offset: 0, profile: 'default' }]
  ])
  expect(Object.values($canonicalGroupBindings.get())).toEqual([{ ...route, roomId: 'retained' }])
})

it.each([false, true])('refreshes discovery after reconnect and refuses the old socket response (batched: %s)', async batched => {
  let releaseOld!: (page: unknown) => void
  let lists = 0
  request.mockImplementation(async method => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}

    if (++lists === 1) {return new Promise(resolve => { releaseOld = resolve })}

    return { rooms: [{ room_id: 'current', name: 'Current room', members: [] }], next_offset: null }
  })
  changeSocket('open')
  render(<CanonicalGroupList onOpen={vi.fn()} />)
  await waitFor(() => expect(lists).toBe(1))
  const epoch = gatewayActivationEpoch()
  const oldPage = { rooms: [{ room_id: 'stale', name: 'Old socket room', members: [] }], next_offset: null }

  if (batched) {
    await act(async () => {
      changeSocket('closed')
      changeSocket('open')
      // Return the old request before React has committed the batched
      // reconnect, so an effect-cleanup flag alone cannot fence it.
      releaseOld(oldPage)
    })
  } else {
    await act(async () => { changeSocket('closed') })
    await act(async () => { changeSocket('open') })
    await screen.findByRole('button', { name: 'Current room' })
    await act(async () => { releaseOld(oldPage) })
  }

  await screen.findByRole('button', { name: 'Current room' })

  expect(gatewayActivationEpoch()).toBe(epoch)
  expect(screen.queryByRole('button', { name: 'Old socket room' })).toBeNull()
  expect(Object.values($canonicalGroupBindings.get()).map(binding => binding.roomId)).toEqual(['current'])
  expect(request.mock.calls.filter(call => call[0] === 'groups.capabilities')).toHaveLength(2)
})

it('rediscovers when startup activates the same open primary while the room list is pending', async () => {
  let releaseFirst!: (page: unknown) => void
  let lists = 0
  const page = { rooms: [{ room_id: 'retained', name: 'Retained room', members: [] }], next_offset: null }
  request.mockImplementation(async method => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}

    return ++lists === 1 ? new Promise(resolve => { releaseFirst = resolve }) : page
  })
  changeSocket('open')
  render(<CanonicalGroupList onOpen={vi.fn()} />)
  await waitFor(() => expect(lists).toBe(1))
  const epoch = gatewayActivationEpoch()
  await act(async () => {
    await ensureGatewayForProfile('default')
    releaseFirst(page)
  })

  expect(gatewayActivationEpoch()).toBeGreaterThan(epoch)
  expect(host.state.connectionId.get()).toBe(connectionId)
  expect(host.state.profile.get()).toBe('default')
  expect(host.state.gateway.get()).toBe('open')
  await screen.findByRole('button', { name: 'Retained room' })
  expect(lists).toBe(2)
  expect(Object.values($canonicalGroupBindings.get()).map(binding => binding.roomId)).toEqual(['retained'])
})

it('removes a confirmed disband from the sidebar immediately without fetching or removing another room', async () => {
  request.mockImplementation(async method => method === 'groups.capabilities' ? CANONICAL_GROUP_CAPABILITIES : {
    rooms: [{ room_id: 'ended', name: 'Ended room', members: [] }, { room_id: 'kept', name: 'Kept room', members: [] }], next_offset: null
  })
  changeSocket('open')
  render(<CanonicalGroupList onOpen={vi.fn()} />)
  await screen.findByRole('button', { name: 'Ended room' })
  const calls = request.mock.calls.length
  await act(async () => { forgetCanonicalGroup({ connectionId, profile: 'default', roomId: 'ended' }) })

  expect(screen.queryByRole('button', { name: 'Ended room' })).toBeNull()
  expect(screen.getByRole('button', { name: 'Kept room' })).toBeTruthy()
  expect(request.mock.calls).toHaveLength(calls)
})

it.each([
  { surface: 'canonical', capability: CANONICAL_GROUP_CAPABILITIES, allowed: true },
  { surface: 'legacy', capability: STANDALONE_GROUP_CAPABILITIES, allowed: false },
  { surface: 'unavailable', capability: { ...CANONICAL_GROUP_CAPABILITIES, driver: false }, allowed: false },
  { surface: 'unadvertised create', capability: { ...CANONICAL_GROUP_CAPABILITIES,
    methods: CANONICAL_GROUP_CAPABILITIES.methods.filter(method => method !== 'groups.create') }, allowed: false }
])('classifies the $surface surface for a fresh Start after the same open primary is reactivated', async ({ capability, allowed }) => {
  let current: unknown = CANONICAL_GROUP_CAPABILITIES
  request.mockImplementation(async (method, params) => method === 'groups.capabilities'
    ? current : { room: { ...params, room_id: 'created' } })
  vi.spyOn(host, 'openWorkspace').mockReturnValue(() => undefined)
  changeSocket('open')
  render(<GroupChatWorkspace group="Classic room" members={[
    { name: 'one', connectionId }, { name: 'two', connectionId }
  ]} />)
  await waitFor(() => expect((screen.getByRole('button', { name: 'Start gateway group' }) as HTMLButtonElement).disabled).toBe(false))
  await act(async () => { await ensureGatewayForProfile('default') })
  current = capability
  fireEvent.click(screen.getByRole('button', { name: 'Start gateway group' }))
  await waitFor(() => expect(request.mock.calls.filter(call => call[0] === 'groups.capabilities')).toHaveLength(2))
  expect(request.mock.calls.filter(call => call[0] === 'groups.create')).toHaveLength(0)

  if (allowed) {
    await waitFor(() => expect((screen.getByRole('button', { name: 'Start gateway group' }) as HTMLButtonElement).disabled).toBe(false))
    fireEvent.click(screen.getByRole('button', { name: 'Start gateway group' }))
    await waitFor(() => expect(request.mock.calls.filter(call => call[0] === 'groups.create')).toHaveLength(1))
    expect(Object.values($canonicalGroupBindings.get())).toEqual([{ connectionId, profile: 'default', roomId: 'created' }])
  } else {
    expect(request.mock.calls.filter(call => call[0] === 'groups.create')).toHaveLength(0)
    expect($canonicalGroupBindings.get()).toEqual({})
  }

  expect(request.mock.calls.filter(call => call[0] === 'groups.capabilities')).toHaveLength(allowed ? 3 : 2)
})

it('reclassifies a settled classic surface after a batched reconnect reports canonical support', async () => {
  let capability: unknown = STANDALONE_GROUP_CAPABILITIES
  request.mockImplementation(async method => method === 'groups.capabilities' ? capability : {})
  changeSocket('open')
  render(<GroupChatWorkspace group="Classic room" members={[
    { name: 'one', connectionId }, { name: 'two', connectionId }
  ]} />)
  await screen.findByRole('textbox')
  expect(request.mock.calls.filter(call => call[0] === 'groups.capabilities')).toHaveLength(1)
  const epoch = gatewayActivationEpoch()
  capability = CANONICAL_GROUP_CAPABILITIES
  await act(async () => { changeSocket('closed'); changeSocket('open') })

  await waitFor(() => expect((screen.getByRole('button', { name: 'Start gateway group' }) as HTMLButtonElement).disabled).toBe(false))
  expect(gatewayActivationEpoch()).toBe(epoch)
  expect(screen.queryByRole('textbox')).toBeNull()
  expect(request.mock.calls.filter(call => call[0] === 'groups.capabilities')).toHaveLength(2)
})

it('does not create from a Start capability reply belonging to the socket that just reconnected', async () => {
  let reads = 0
  let releaseProbe!: (capability: unknown) => void
  request.mockImplementation(async method => {
    if (method !== 'groups.capabilities') {return { room: { room_id: 'wrong' } }}

    return ++reads === 2 ? new Promise(resolve => { releaseProbe = resolve }) : CANONICAL_GROUP_CAPABILITIES
  })
  changeSocket('open')
  render(<GroupChatWorkspace group="Classic room" members={[
    { name: 'one', connectionId }, { name: 'two', connectionId }
  ]} />)
  await waitFor(() => expect((screen.getByRole('button', { name: 'Start gateway group' }) as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'Start gateway group' }))
  await waitFor(() => expect(reads).toBe(2))
  const epoch = gatewayActivationEpoch()
  await act(async () => {
    changeSocket('closed')
    changeSocket('open')
    releaseProbe(CANONICAL_GROUP_CAPABILITIES)
  })

  expect(gatewayActivationEpoch()).toBe(epoch)
  expect(request.mock.calls.filter(call => call[0] === 'groups.create')).toHaveLength(0)
  expect($canonicalGroupBindings.get()).toEqual({})
})

it('keeps the new socket classification when an old rendered Start refreshes after the reconnect', async () => {
  let reads = 0
  let releaseProbe!: (capability: unknown) => void
  request.mockImplementation(async method => {
    if (method !== 'groups.capabilities') {return { room: { room_id: 'wrong' } }}

    return ++reads === 2 ? new Promise(resolve => { releaseProbe = resolve }) : CANONICAL_GROUP_CAPABILITIES
  })
  changeSocket('open')
  render(<GroupChatWorkspace group="Classic room" members={[
    { name: 'one', connectionId }, { name: 'two', connectionId }
  ]} />)
  const button = await screen.findByRole('button', { name: 'Start gateway group' })
  await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false))
  await act(async () => {
    changeSocket('closed')
    changeSocket('open')
    fireEvent.click(button)
  })
  await waitFor(() => expect(reads).toBe(3))
  await act(async () => { releaseProbe(CANONICAL_GROUP_CAPABILITIES) })

  await waitFor(() => expect((screen.getByRole('button', { name: 'Start gateway group' }) as HTMLButtonElement).disabled).toBe(false))
  expect(request.mock.calls.filter(call => call[0] === 'groups.create')).toHaveLength(0)
  expect($canonicalGroupBindings.get()).toEqual({})
})
