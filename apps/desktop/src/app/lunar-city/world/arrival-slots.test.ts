import { expect, it, vi } from 'vitest'

import type { EntityKey, LunarCitySnapshot, Vec3 } from '../model'

import { createArrivalSlots } from './arrival-slots'
import { createNavigationController } from './navigation'

const origin = { x: 0, y: 0, z: 0 }
const manifest = { navigation: { links: [{ from: origin, to: { x: 4, y: .4, z: 0 }, bidirectional: true }] } }
const key = (id: string) => id as EntityKey

const snapshot = (ids: string[]): LunarCitySnapshot => ({ revision: 1, observedAt: 1, sources: [], entities: new Map(ids.map(id => {
  const k = key(id)

  return [k, { key: k, identity: { kind: 'session', connectionId: 'local', profile: 'p', sessionId: id }, authority: 'authoritative', destination: 'review', observedAt: 1, animation: 'review' }]
})) })

it('keeps distinct approach slots stable across source polling and never collapses short-route rows', () => {
  const slots = createArrivalSlots(manifest)
  const initial = ['a', 'b', 'c', 'd'].map(id => slots.target(key(id), 'review', origin))
  expect(new Set(initial.map(point => JSON.stringify(point))).size).toBe(4)
  slots.update(snapshot(['d', 'b', 'a', 'c']))
  expect(['a', 'b', 'c', 'd'].map(id => slots.target(key(id), 'review', origin))).toEqual(initial)
  // This four-metre approach has room for two rows; excess workers use the declared anchor.
  expect(slots.target(key('overflow'), 'review', origin)).toEqual(origin)
  slots.update(snapshot(['a', 'c', 'd']))
  expect(slots.target(key('replacement'), 'review', origin)).toEqual(initial[1])
  expect(slots.target(key('d'), 'review', origin)).toEqual(initial[3])
})

it('pause holds physical positions while preserving distinct arrival slots for resumed routes', () => {
  const slots = createArrivalSlots(manifest)
  const query = { computePath: vi.fn((from: Vec3, to: Vec3) => [from, to]) }

  const navigation = createNavigationController({ destinations: { review: origin }, query, targetFor: slots.target,
    workerClips: new Set(['walk', 'idle', 'review']), speedUnitsPerSecond: 1 })

  const a = { key: key('a'), position: { x: -4, y: 0, z: 0 }, animation: 'review' }
  const b = { key: key('b'), position: { x: -4, y: 0, z: 0 }, animation: 'review' }
  navigation.move(a, 'review')
  navigation.tick(100)
  slots.update(snapshot(['a', 'b']))
  navigation.move(b, 'review')
  const aTarget = slots.target(a.key, 'review', origin), bTarget = slots.target(b.key, 'review', origin)
  const heldA = { ...a.position }, heldB = { ...b.position }
  navigation.setReducedMotion(true)
  expect(a.position).toEqual(heldA)
  expect(b.position).toEqual(heldB)
  expect(a.position).not.toEqual(b.position)
  expect(navigation.isMoving(a.key)).toBe(false)
  slots.update(snapshot(['b', 'a']))
  navigation.move(a, 'review')
  expect(a.position).toEqual(heldA)
  expect(a.animation).toBe('idle')
  expect(slots.target(a.key, 'review', origin)).toEqual(aTarget)
  expect(slots.target(b.key, 'review', origin)).toEqual(bTarget)
  navigation.setReducedMotion(false)
  navigation.move(a, 'review')
  navigation.move(b, 'review')

  for (let frame = 0; frame < 30; frame++) {navigation.tick(250)}
  expect(a.position).toEqual(aTarget)
  expect(b.position).toEqual(bTarget)
})
