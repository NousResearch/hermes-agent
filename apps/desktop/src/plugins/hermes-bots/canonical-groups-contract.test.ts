import { beforeEach, expect, it, vi } from 'vitest'

import type { captureGroupRequests } from './canonical-group-contract-test-utils'

const doubles = vi.hoisted(() => ({ request: vi.fn(), capture: undefined as ReturnType<typeof captureGroupRequests> | undefined }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { captureGroupRequests } = await import('./canonical-group-contract-test-utils')
  doubles.capture = captureGroupRequests(doubles.request)

  return { host: { requestProfile: doubles.capture.request } }
})

import {
  assertCanonicalGroupCalls,
  CANONICAL_GROUP_CAPABILITIES,
  declaredCapabilities,
  declaredGroupMethod
} from './canonical-group-contract-test-utils'
import { actCanonicalGroup, canonicalGroupRequest, createCanonicalGroup, discoverCanonicalGroups } from './canonical-groups'

const route = { connectionId: 'owner', profile: 'team' }
beforeEach(() => doubles.request.mockReset())

it('reads the generated gateway contract it checks against', () => {
  expect(declaredGroupMethod('groups.send')?.params).toEqual({ fields: ['profile', 'room_id', 'event_id', 'payload'], required: ['room_id', 'payload'] })
  expect(declaredGroupMethod('groups.capabilities')?.params).toEqual({ fields: ['profile'], required: [] })
  // The gateway's own declaration wins over the standalone server's (which takes only room_id and task_id).
  expect(declaredGroupMethod('groups.retry')?.params.required).toEqual(['room_id', 'member_id', 'task_id', 'execution_generation'])
  expect(declaredGroupMethod('groups.not_declared')).toBeUndefined()
  expect(() => declaredCapabilities({ ...CANONICAL_GROUP_CAPABILITIES, methods: ['groups.not_declared'], extra: true }))
    .toThrow('does not fit the gateway contract: groups.not_declared, extra')
})

it('checks recorded discovery, creation and exact attempt calls against the gateway contract', async () => {
  doubles.request.mockResolvedValueOnce(CANONICAL_GROUP_CAPABILITIES).mockResolvedValueOnce({ rooms: [], next_offset: null })
  await discoverCanonicalGroups(route)
  doubles.request.mockImplementation(async (_route, _method, params) => ({ room: { ...params } }))
  const { binding } = await createCanonicalGroup(route, 'Team', [{ name: 'alice' }, { name: 'bob' }])

  for (const kind of ['retry', 'discard', 'approval']) {
    await actCanonicalGroup(binding, { kind, member_id: 'alice', task_id: 'task', execution_generation: 7, request_id: 'approval' }, 'deny')
  }

  const calls = doubles.capture!.calls
  expect(calls.map(call => call.method)).toEqual(['groups.capabilities', 'groups.list', 'groups.create', 'groups.retry', 'groups.discard', 'groups.approve'])
  expect(calls.every(call => call.params.profile === route.profile)).toBe(true)
  expect(() => assertCanonicalGroupCalls(calls)).not.toThrow()
})

it('rejects a captured undeclared method, unknown field or missing field even when the request double accepts it', async () => {
  await canonicalGroupRequest(route, 'groups.not_declared')
  await canonicalGroupRequest(route, 'groups.state', { room_id: 'room', undeclared: true })
  // The standalone server declares groups.promote; the gateway does not serve it.
  await canonicalGroupRequest(route, 'groups.promote', { room_id: 'room', confirm: true })
  await canonicalGroupRequest(route, 'groups.send', { room_id: 'room', event_id: 'event', payload: undefined })
  const calls = doubles.capture!.calls.splice(0)
  expect(calls.slice(0, 2)).toEqual([
    { method: 'groups.not_declared', params: { profile: route.profile } },
    { method: 'groups.state', params: { room_id: 'room', undeclared: true, profile: route.profile } }
  ])
  expect(() => assertCanonicalGroupCalls([calls[0]])).toThrow('Undeclared canonical group method: groups.not_declared')
  expect(() => assertCanonicalGroupCalls([calls[1]])).toThrow('Undeclared canonical group field: groups.state.undeclared')
  expect(() => assertCanonicalGroupCalls([calls[2]])).toThrow('Undeclared canonical group method: groups.promote')
  expect(() => assertCanonicalGroupCalls([calls[3]])).toThrow('Missing canonical group field: groups.send.payload')
})
