import { expect, it } from 'vitest'

import type { EntityKey } from '../model'

import { createEntityPositionState } from './entity-position-state'

it('scopes opaque identities, copies retained coordinates, and restores across handle recreation', () => {
  const key = 'same-opaque-key' as EntityKey
  const saved = { position: { x: 4, y: 0, z: 8 }, sourcePosition: { x: 1, y: 0, z: 2 } }
  const state = createEntityPositionState({ scope: 'world-a/nav-a' })
  state.save(key, saved, true)
  saved.position.x = 99
  const restored = createEntityPositionState({ scope: 'world-a/nav-a' }).read(key)!
  expect(restored.position.x).toBe(4)
  restored.position.x = 50
  expect(state.read(key)?.position.x).toBe(4)
  expect(createEntityPositionState({ scope: 'world-a/nav-b' }).read(key)).toBeUndefined()
  state.forget(key)
  expect(state.read(key)).toBeUndefined()
})

it('bounds retained workers and expires records without renewing leases for motion-only writes', () => {
  let time = 0
  const state = createEntityPositionState({ scope: 'bounded', maxEntries: 2, ttlMs: 10, now: () => time })
  const key = (name: string) => name as EntityKey
  const value = { position: { x: 0, y: 0, z: 0 } }

  for (const name of ['one', 'two', 'three']) {state.save(key(name), value, true)}
  expect(state.read(key('one'))).toBeUndefined()
  time = 9
  state.save(key('two'), { position: { x: 5, y: 0, z: 0 } }, false)
  expect(state.read(key('two'))?.position.x).toBe(5)
  time = 11
  state.save(key('two'), value, false)
  expect(state.read(key('two'))).toBeUndefined()
  expect(state.read(key('three'))).toBeUndefined()
})
