import { beforeEach, expect, it, vi } from 'vitest'

const host = vi.hoisted(() => ({
  requestProfile: vi.fn(),
  request: vi.fn(),
  state: { connectionId: { get: vi.fn() }, profile: { get: vi.fn() } }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { captureGroupRequests } = await import('./group-test-utils')

  return { host: { ...host, requestProfile: captureGroupRequests(host.requestProfile).request } }
})

import { groupExecutionMode } from './canonical-group-capabilities'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { actCanonicalGroup, canonicalGroupCreateErrorMessage, canonicalGroupEligibility, canonicalGroupRequest, captureCanonicalGroupRoute, createCanonicalGroup, discoverCanonicalGroups, isCanonicalGroupCreateRefusal } from './canonical-groups'
import { CANONICAL_GROUP_CAPABILITIES, STANDALONE_GROUP_CAPABILITIES } from './group-test-utils'

it('formats local eligibility and generic gateway refusals in the caller’s locale without diagnosing profile setup', async () => {
  const error = await createCanonicalGroup({ connectionId: 'source-a', profile: 'default' }, 'Team', [{ name: 'alice' }])
    .catch(failure => failure)

  expect(canonicalGroupCreateErrorMessage(error, CANONICAL_GROUP_LOCALES.fr)).toBe(CANONICAL_GROUP_LOCALES.fr.classicCount)
  expect(host.requestProfile).not.toHaveBeenCalled()
  const refused = Object.assign(new Error('invalid_params'), { code: 4001, data: { reason: 'invalid_params' } })

  expect(canonicalGroupCreateErrorMessage(refused, CANONICAL_GROUP_LOCALES.fr)).toBe(CANONICAL_GROUP_LOCALES.fr.createRefused)
  expect(canonicalGroupCreateErrorMessage(refused, CANONICAL_GROUP_LOCALES.fr)).not.toContain('hosted_rooms.profiles')
})

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
  host.requestProfile.mockResolvedValueOnce(CANONICAL_GROUP_CAPABILITIES)
    .mockResolvedValueOnce({ rooms: [], next_offset: 100 })
    .mockResolvedValueOnce({ rooms: [room], next_offset: null })
  expect(await discoverCanonicalGroups(route)).toEqual({ driver: true, rooms: [room] })
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
  host.requestProfile.mockResolvedValueOnce(CANONICAL_GROUP_CAPABILITIES).mockResolvedValueOnce({ rooms: [], next_offset: 0 })
  await expect(discoverCanonicalGroups(route)).rejects.toThrow('pagination')
  expect(host.request).not.toHaveBeenCalled()
})

it('classifies the advertised surface and discovers rooms only for a live canonical driver', async () => {
  const rows = [
    [CANONICAL_GROUP_CAPABILITIES, undefined, 'canonical'],
    [STANDALONE_GROUP_CAPABILITIES, undefined, 'legacy'],
    [{ ...CANONICAL_GROUP_CAPABILITIES, driver: false }, undefined, 'unavailable'],
    [{ ...CANONICAL_GROUP_CAPABILITIES, driver: undefined }, undefined, 'unavailable'],
    [{ ...CANONICAL_GROUP_CAPABILITIES, driver: 'true' }, undefined, 'unavailable'],
    [{ methods: [] }, undefined, 'legacy'],
    [{ driver: true }, undefined, 'unavailable'],
    [{ methods: 'groups.discard', driver: true }, undefined, 'unavailable'],
    [{ methods: ['groups.discard', null], driver: true }, undefined, 'unavailable'],
    [null, undefined, 'unavailable'],
    [undefined, { code: -32601 }, 'legacy'],
    [undefined, new Error('timeout'), 'unavailable'],
    [undefined, new Error('transport disconnected'), 'unavailable'],
    [undefined, { code: 401 }, 'unavailable']
  ] as const

  const classified = []
  const discovered = []
  const wire = []

  for (const [index, [value, error]] of rows.entries()) {
    classified.push(groupExecutionMode(value, error))
    host.requestProfile.mockClear().mockImplementation(async (_route, method) => {
      if (method === 'groups.capabilities') {
        if (error) {throw error}

        return value
      }

      return { rooms: [{ room_id: 'canonical-room', name: 'Team', members: [] }], next_offset: null }
    })
    discovered.push(await discoverCanonicalGroups({ connectionId: 'table-owner', profile: `profile-${index}` }))
    wire.push(host.requestProfile.mock.calls.map(call => call[1]))
  }

  expect(classified).toEqual(rows.map(row => row[2]))
  expect(discovered.map(result => [result.driver, result.rooms.length])).toEqual(rows.map(row => [row[2] === 'canonical', row[2] === 'canonical' ? 1 : 0]))
  expect(wire).toEqual(rows.map(row => row[2] === 'canonical' ? ['groups.capabilities', 'groups.list'] : ['groups.capabilities']))
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
  expect(canonicalGroupEligibility(route, members)).toEqual({ eligible: true, roster: room.members })
  const six = Array.from({ length: 6 }, (_, index) => ({ name: `member-${index}`, connectionId: route.connectionId }))
  expect(canonicalGroupEligibility(route, six).eligible).toBe(true)
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
    [members[0]], [members[0], members[0]], [...six, { name: 'seventh' }],
    [{ ...members[0], handle: 'ALL' }, members[1]],
    [{ ...members[0], handle: 'everyone' }, members[1]],
    [{ ...members[0], handle: 'BOB' }, members[1]],
    [{ ...members[0], targetProfile: 'BOB' }, members[1]],
    [{ ...members[0], handle: ' ' }, members[1]]
  ]) {
    expect(canonicalGroupEligibility(route, invalid).eligible).toBe(false)
    await expect(createCanonicalGroup(route, 'Team', invalid)).rejects.toThrow()
  }

  expect(host.requestProfile).toHaveBeenCalledTimes(1)
  host.requestProfile.mockClear()
  const identity = { member_id: 'alice', task_id: 'task:original', execution_generation: 7 }

  for (const kind of ['retry', 'discard']) {
    await actCanonicalGroup(binding, { kind, ...identity, request_id: 'not-for-this-method' })
    expect(host.requestProfile).toHaveBeenLastCalledWith(expect.objectContaining(route), `groups.${kind}`, {
      profile: route.profile, room_id: binding.roomId, ...identity
    })
  }

  await actCanonicalGroup(binding, { kind: 'approval', ...identity, request_id: 'request:original', approval: { choices: ['deny'] } }, 'deny')
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

it('recognizes only typed invalid-params creation refusals, not transport or unrelated errors', () => {
  expect(isCanonicalGroupCreateRefusal({ code: 4001, data: { reason: 'invalid_params' } })).toBe(true)

  for (const error of [null, new Error('invalid_params'), { code: 4001 },
    { code: 4001, data: { reason: 'not_ready' } }, { code: 401, data: { reason: 'invalid_params' } }]) {
    expect(isCanonicalGroupCreateRefusal(error)).toBe(false)
  }
})
