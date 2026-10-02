import { beforeEach, expect, it, vi } from 'vitest'

import type { captureGroupRequests } from './group-test-utils'

const doubles = vi.hoisted(() => ({ request: vi.fn(), capture: undefined as ReturnType<typeof captureGroupRequests> | undefined }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { captureGroupRequests } = await import('./group-test-utils')
  doubles.capture = captureGroupRequests(doubles.request)

  return { host: { requestProfile: doubles.capture.request } }
})

import { actCanonicalGroup, canonicalGroupRequest, createCanonicalGroup, discoverCanonicalGroups } from './canonical-groups'
import { assertCanonicalGroupCalls, CANONICAL_GROUP_CAPABILITIES } from './group-test-utils'

const route = { connectionId: 'owner', profile: 'team' }
beforeEach(() => doubles.request.mockReset())

it('checks recorded discovery, creation and exact attempt calls against the canonical wire contract', async () => {
  doubles.request.mockResolvedValueOnce(CANONICAL_GROUP_CAPABILITIES).mockResolvedValueOnce({ rooms: [], next_offset: null })
  await discoverCanonicalGroups(route)
  doubles.request.mockImplementation(async (_route, _method, params) => ({ room: { ...params } }))
  const { binding } = await createCanonicalGroup(route, 'Team', [{ name: 'alice' }, { name: 'bob' }])

  for (const kind of ['retry', 'discard', 'approval']) {
    await actCanonicalGroup(binding, { kind, member_id: 'alice', task_id: 'task', execution_generation: 7, request_id: 'approval', approval: { choices: ['deny'] } }, 'deny')
  }

  const calls = doubles.capture!.calls
  expect(calls.map(call => call.method)).toEqual(['groups.capabilities', 'groups.list', 'groups.create', 'groups.retry', 'groups.discard', 'groups.approve'])
  expect(calls.every(call => call.params.profile === route.profile)).toBe(true)
  expect(() => assertCanonicalGroupCalls(calls)).not.toThrow()
  const params = { include_sessions: true }
  await doubles.capture!.request(route, 'profiles.list', params, 123)
  expect(doubles.request).toHaveBeenLastCalledWith(route, 'profiles.list', params, 123)
  expect(calls).toHaveLength(6)
})

it('rejects a captured undeclared method or field even when the request double accepts it', async () => {
  await canonicalGroupRequest(route, 'groups.not_declared')
  await canonicalGroupRequest(route, 'groups.state', { room_id: 'room', undeclared: true })
  const calls = doubles.capture!.calls.splice(0)
  expect(calls).toEqual([
    { method: 'groups.not_declared', params: { profile: route.profile } },
    { method: 'groups.state', params: { room_id: 'room', undeclared: true, profile: route.profile } }
  ])
  expect(() => assertCanonicalGroupCalls([calls[0]])).toThrow('Undeclared canonical group method: groups.not_declared')
  expect(() => assertCanonicalGroupCalls([calls[1]])).toThrow('Undeclared canonical group field: groups.state.undeclared')
})
