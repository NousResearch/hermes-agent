import { expect, it } from 'vitest'

import { entityKey } from '../identity'
import type { LunarCitySnapshot, LunarEntity } from '../model'

import { createWorkerCompletionController } from './worker-completion'

const identity = { kind: 'session', connectionId: 'local', profile: 'p', sessionId: 's' } as const
const key = entityKey(identity)
const worker = (sourceState: string, observedAt: number): LunarEntity => ({ key, identity, sourceState, observedAt, animation: 'done', authority: 'authoritative', destination: 'project' })
const snapshot = (...entities: LunarEntity[]): LunarCitySnapshot => ({ entities: new Map(entities.map(entity => [entity.key, entity])), observedAt: 1, revision: 1, sources: [] })
const env = { enabled: true, available: () => true, moving: () => false, supports: (_key: unknown, clip: string) => clip === 'handoff' }

it('requires a strictly newer authoritative completion and never replays history, stale recovery or existing finite clips', () => {
  const controller = createWorkerCompletionController()
  controller.update(snapshot(worker('done', 1)))
  expect(controller.tick(0, env).size).toBe(0)
  controller.update(snapshot(worker('running', 2)))
  controller.update(snapshot(worker('done', 2)))
  expect(controller.tick(0, env).size).toBe(0)
  controller.update(snapshot(worker('running', 3)))
  controller.update(snapshot(worker('done', 4)))
  expect(controller.tick(0, env).get(key)?.animation).toBe('handoff')
  controller.update(snapshot(worker('running', 3)))
  controller.update(snapshot(worker('done', 4)))
  expect(controller.tick(3100, env).size).toBe(0)
  controller.update(snapshot(worker('done', 5)))
  expect(controller.tick(0, env).size).toBe(0)
  controller.update(snapshot({ ...worker('running', 6), authority: 'stale' }))
  controller.update(snapshot(worker('done', 7)))
  expect(controller.tick(0, env).size).toBe(0)
  controller.update(snapshot(worker('running', 8)))
  controller.update(snapshot(worker('done', 9)))
  expect(controller.tick(0, { ...env, supports: () => true }).size).toBe(0)
})

it('uses only actual fallback motion, yields to movement/work and clears when disabled or removed', () => {
  const controller = createWorkerCompletionController()
  let stamp = 0

  const queue = () => {
    controller.update(snapshot(worker('running', ++stamp)))
    const done = worker('done', ++stamp)
    controller.update(snapshot(done))

    return done
  }

  const done = queue()
  const before = JSON.stringify(done)
  expect(controller.tick(0, { ...env, supports: (_key, clip) => clip === 'celebrate' }).get(key)?.animation).toBe('celebrate')
  expect(JSON.stringify(done)).toBe(before)
  expect(controller.tick(10, { ...env, moving: () => true }).size).toBe(0)
  expect(controller.tick(10, env).size).toBe(0)
  queue()
  expect(controller.tick(0, { ...env, supports: () => false }).size).toBe(0)
  queue()
  expect(controller.tick(0, { ...env, enabled: false }).size).toBe(0)
  expect(controller.tick(0, env).size).toBe(0)
  queue()
  controller.update(snapshot(worker('working', ++stamp)))
  expect(controller.tick(0, env).size).toBe(0)
  queue()
  controller.update(snapshot())
  expect(controller.isActive()).toBe(false)
})
