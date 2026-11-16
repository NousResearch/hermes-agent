import { beforeEach, expect, it } from 'vitest'

import {
  $canonicalGroupBindings,
  canonicalGroupAliasKey,
  canonicalGroupBindingFor,
  canonicalGroupKeyRoute,
  publishCanonicalGroup
} from './canonical-group-registry'

// Independent review on #140972 (#140805): room names are free text (hosted_rooms
// has no pattern), so a room on connection B named like a `canonical:` key parsed
// as A's key and opened A's room. A NAME is never a key; a KEY keeps main's format.
const A = { connectionId: 'conn-A', profile: 'work' }
const B = { connectionId: 'conn-B', profile: 'work' }
const bind = (route: typeof A, roomId: string) => ({ ...route, roomId, roomName: 'Team' })

beforeEach(() => {
  $canonicalGroupBindings.set({})
})

it('a room NAME shaped like a canonical key is not treated as an identity on another connection', () => {
  const name = 'canonical:conn-A:work:ra'
  const bindings = { [name]: bind(A, 'ra') }
  expect(canonicalGroupKeyRoute(name, 'name')).toBeNull()
  expect(canonicalGroupBindingFor(bindings, B, name, 'name')).toBeUndefined()
})

it('a main-format key minted before this change still opens its room', () => {
  // A restored tab / a key minted by main or an older build: canonical:<conn>:<profile>:<room>.
  const key = 'canonical:conn-A:work:ra'
  expect(canonicalGroupKeyRoute(key, 'key')).toEqual({ ...A, roomId: 'ra' })
  expect(canonicalGroupBindingFor({ [key]: bind(A, 'ra') }, B, key, 'key')?.roomId).toBe('ra')
})

it('a canonical-shaped name resolved on B is published under B alias, never as a raw key', () => {
  const name = 'canonical:conn-A:work:ra'
  publishCanonicalGroup(B, name, { authority: B, room: { room_id: 'rb', name, members: [] } }, 'name')
  const bindings = $canonicalGroupBindings.get()
  expect(bindings[name]).toBeUndefined()
  expect(canonicalGroupBindingFor(bindings, B, name, 'name')?.roomId).toBe('rb')
  expect(canonicalGroupBindingFor(bindings, A, name, 'name')).toBeUndefined()
})

it('alias keys stay unambiguous when the name itself carries separators', () => {
  expect(canonicalGroupAliasKey({ connectionId: 'a', profile: 'b' }, 'c:d')).not.toBe(
    canonicalGroupAliasKey({ connectionId: 'a', profile: 'b:c' }, 'd')
  )
})
