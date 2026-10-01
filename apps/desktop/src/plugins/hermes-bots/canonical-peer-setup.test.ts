import { afterEach, expect, test, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', () => ({ host: {} }))

import { canonicalPeerGroupEligibility, createCanonicalPeerGroup } from './canonical-groups'

const route = { connectionId: 'home', profile: 'default' }
const members = [
  { name: 'default', handle: 'home', connectionId: 'home' },
  { name: 'default', handle: 'peer', connectionId: 'peer' }
]
const desktop = window.hermesDesktop
afterEach(() => {window.hermesDesktop = desktop})

test('mixed gateways preserve member identities and send only intent through native IPC', async () => {
  const room = { room_id: 'created', name: 'Room', members: [] }
  const create = vi.fn(async () => ({ ok: true, room }))
  window.hermesDesktop = { roomSetup: { create, recover: vi.fn() } } as any
  expect(canonicalPeerGroupEligibility(route, members)).toBe(true)
  expect(canonicalPeerGroupEligibility(route, [members[0], { ...members[1], targetProfile: 'private' }])).toBe(false)
  const result = await createCanonicalPeerGroup(route, 'Room', members)
  expect(result.binding).toEqual({ ...route, roomId: room.room_id })
  expect(create).toHaveBeenCalledWith({ home: route, name: 'Room', members: [
    { member_id: 'member-1', handle: 'home', connectionId: 'home', profile: 'default' },
    { member_id: 'member-2', handle: 'peer', connectionId: 'peer', profile: 'default' }
  ] })
  create.mockResolvedValueOnce({ ok: false, reason: 'cleanup_pending' } as any)
  await expect(createCanonicalPeerGroup(route, 'Room', members)).rejects.toMatchObject({ roomSetupReason: 'cleanup_pending' })
})
