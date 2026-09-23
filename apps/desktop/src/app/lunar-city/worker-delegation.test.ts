import { expect, it } from 'vitest'

import { entityKey } from './identity'
import type { LunarCitySnapshot, LunarEntity } from './model'
import { createWorkerDelegationController } from './worker-delegation'

function pair(connectionId = 'local') {
  const identity = { kind: 'session', connectionId, profile: 'p', sessionId: 's' } as const

  const parent: LunarEntity = {
    key: entityKey(identity),
    identity,
    authority: 'authoritative',
    observedAt: 1,
    animation: 'work',
    sourceState: 'running',
    destination: 'project'
  }

  const childIdentity = { ...identity, kind: 'subagent', subagentId: 'c' } as const

  return [parent, { ...parent, identity: childIdentity, key: entityKey(childIdentity) }] as const
}

const snapshot = (entities: readonly LunarEntity[]): LunarCitySnapshot => ({
  revision: 1,
  observedAt: 1,
  sources: [],
  entities: new Map(entities.map(entity => [entity.key, entity]))
})

const env = {
  presentation: () => ({ position: { x: 1, y: 2, z: 3 }, moving: true, animation: 'walk' }),
  project: () => ({ x: 10, y: 20 })
}

it('requires exact authoritative parent identity and actual paired movement or explicit handoff state', () => {
  const [parent, child] = pair(),
    [foreign] = pair('remote')

  const controller = createWorkerDelegationController()
  controller.update(snapshot([foreign, child]))
  expect(controller.links(0, env)).toEqual([])
  controller.update(snapshot([parent, child]))
  expect(controller.links(0, env)[0]?.kind).toBe('traveling')
  const resting = { ...env, presentation: () => ({ ...env.presentation(), moving: false }) }
  expect(controller.links(0, resting)).toEqual([])
  controller.update(snapshot([parent, { ...child, sourceState: 'dependency', observedAt: 2 }]))
  expect(controller.links(0, resting)[0]?.kind).toBe('handoff')
  expect(controller.links(3100, resting)).toEqual([])
  controller.update(snapshot([parent, { ...child, sourceState: 'dependency', observedAt: 3 }]))
  expect(controller.links(0, resting)).toEqual([])
  expect(controller.hasCandidates()).toBe(false)
  controller.update(snapshot([{ ...parent, authority: 'stale' }, child]))
  expect(controller.links(0, env)).toEqual([])
})
it('shows finite returns only on newly observed authoritative completion, never initial history or stale recovery', () => {
  const [parent, child] = pair(),
    done = { ...child, sourceState: 'done', observedAt: 2 }

  const controller = createWorkerDelegationController()
  controller.update(snapshot([parent, done]))
  expect(controller.links(0, env)).toEqual([])
  controller.clear()
  controller.update(snapshot([parent, child]))
  controller.update(snapshot([parent, { ...done, observedAt: 1 }]))
  expect(controller.links(0, { ...env, presentation: () => ({ ...env.presentation(), moving: false }) })).toEqual([])
  controller.update(snapshot([parent, done]))
  expect(controller.links(0, env)[0]?.kind).toBe('return')
  controller.update(snapshot([parent, done]))
  expect(controller.links(3100, env)).toEqual([])
  controller.update(snapshot([parent, { ...child, authority: 'stale' }]))
  controller.update(snapshot([parent, done]))
  expect(controller.links(0, env)).toEqual([])
  controller.update(snapshot([]))
  expect(controller.hasCandidates()).toBe(false)
})
