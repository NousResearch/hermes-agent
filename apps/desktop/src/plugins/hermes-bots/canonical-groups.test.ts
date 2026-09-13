import { beforeEach, expect, it, vi } from 'vitest'

const host = vi.hoisted(() => ({
  requestProfile: vi.fn(),
  request: vi.fn(),
  state: { connectionId: { get: vi.fn() }, profile: { get: vi.fn() } }
}))

vi.mock('@hermes/plugin-sdk', () => ({ host }))

import { actCanonicalGroup, canonicalGroupRequest, captureCanonicalGroupRoute, createCanonicalGroup, discoverCanonicalGroups } from './canonical-groups'

beforeEach(() => {
  vi.resetAllMocks()
  host.state.connectionId.get.mockReturnValue('source-a')
  host.state.profile.get.mockReturnValue('default')
})

it('pins discovery and every subsequent request to its captured authority, including empty filtered pages', async () => {
  const route = captureCanonicalGroupRoute()
  const room = { room_id: 'room-a', name: 'Team', members: [] }
  host.state.connectionId.get.mockReturnValue('source-b')
  host.state.profile.get.mockReturnValue('other')
  host.requestProfile.mockResolvedValueOnce({ driver: false })
    .mockResolvedValueOnce({ rooms: [], next_offset: 100 })
    .mockResolvedValueOnce({ rooms: [room], next_offset: null })
  expect(await discoverCanonicalGroups(route)).toEqual({ driver: false, rooms: [room] })
  expect(host.requestProfile.mock.calls.map(call => [call[1], call[2]])).toEqual([
    ['groups.capabilities', { profile: 'default' }],
    ['groups.list', { profile: 'default', limit: 100, offset: 0 }],
    ['groups.list', { profile: 'default', limit: 100, offset: 100 }]
  ])

  for (const [target] of host.requestProfile.mock.calls) {
    expect(target).toMatchObject({ connectionId: 'source-a', profile: 'default', targetProfile: 'default' })
  }

  host.requestProfile.mockRejectedValueOnce(new Error('owner disconnected'))
  await expect(canonicalGroupRequest(route, 'groups.state', { room_id: room.room_id })).rejects.toThrow('owner disconnected')
  await expect(canonicalGroupRequest({ ...route, connectionId: '' }, 'groups.list', {})).rejects.toThrow()
  await expect(canonicalGroupRequest(route, 'groups.list', { profile: 'other' })).rejects.toThrow()
  host.requestProfile.mockResolvedValueOnce({ driver: true }).mockResolvedValueOnce({ rooms: [], next_offset: 0 })
  await expect(discoverCanonicalGroups(route)).rejects.toThrow('pagination')
  expect(host.request).not.toHaveBeenCalled()
})

it('creates only same-authority rosters and dispatches exact advertised attempt identities without inference', async () => {
  const route = captureCanonicalGroupRoute()

  const members = [
    { name: 'alice', handle: 'alice', connectionId: 'source-a', display_name: 'Alice' },
    { name: 'desktop-bob', handle: 'bob', connectionId: 'source-a', targetProfile: 'bob' }
  ]

  host.requestProfile.mockImplementation(async (_route, method, params) => method === 'groups.create'
    ? { room: { room_id: params.room_id, name: params.name, members: params.members } } : { accepted: true })
  const { binding, room } = await createCanonicalGroup(route, 'Team', members)
  expect(binding).toEqual({ ...route, roomId: room.room_id })
  expect(room.room_id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)
  expect(room.members).toEqual([
    { member_id: 'alice', profile: 'alice', handle: 'alice', display_name: 'Alice', target: { kind: 'local', profile: 'alice' } },
    { member_id: 'bob', profile: 'bob', handle: 'bob', target: { kind: 'local', profile: 'bob' } }
  ])

  for (const invalid of [
    [{ ...members[0], connectionId: 'source-b' }, members[1]],
    [{ ...members[0], remoteSource: true, connectionId: undefined }, members[1]],
    [{ ...members[0], route: { connectionId: 'source-b', profile: 'alice', targetProfile: 'alice', mode: 'remote' as const } }, members[1]],
    [members[0]], [members[0], members[0]]
  ]) {await expect(createCanonicalGroup(route, 'Team', invalid)).rejects.toThrow()}

  expect(host.requestProfile).toHaveBeenCalledTimes(1)
  host.requestProfile.mockClear()
  const identity = { member_id: 'alice', task_id: 'task:original', execution_generation: 7 }

  for (const kind of ['retry', 'discard']) {
    await actCanonicalGroup(binding, { kind, ...identity, request_id: 'not-for-this-method' })
    expect(host.requestProfile).toHaveBeenLastCalledWith(expect.objectContaining(route), `groups.${kind}`, {
      profile: route.profile, room_id: binding.roomId, ...identity
    })
  }

  await actCanonicalGroup(binding, { kind: 'approval', ...identity, request_id: 'request:original' }, 'deny')
  expect(host.requestProfile).toHaveBeenLastCalledWith(expect.objectContaining(route), 'groups.approve', {
    profile: route.profile, room_id: binding.roomId, ...identity, request_id: 'request:original', choice: 'deny'
  })
  const count = host.requestProfile.mock.calls.length

  for (const action of [
    { kind: 'approval', ...identity }, { kind: 'retry', ...identity, execution_generation: 0 },
    { kind: 'unknown', ...identity }, { kind: 'toString', ...identity }, { kind: 'discard', ...identity, task_id: '' }
  ]) {await expect(actCanonicalGroup(binding, action)).rejects.toThrow()}

  expect(host.requestProfile).toHaveBeenCalledTimes(count)
  expect(host.request).not.toHaveBeenCalled()
})
