import { expect, it } from 'vitest'

import { entityKey, projectCompoundKey } from '../identity'
import type { EntityIdentity, LunarCitySnapshot, LunarEntity, ProjectSlotManifestEntry } from '../model'

import { createProjectCompoundController } from './project-compounds'

function slot(id: string, x: number): ProjectSlotManifestEntry {
  return { id, position: { x, y: 0, z: 0 }, bounds: { min: { x: x - 1, y: 0, z: -1 }, max: { x: x + 1, y: 4, z: 1 } }, navigationLink: { from: { x, y: 0, z: 0 }, to: { x, y: 0, z: 2 }, bidirectional: true } }
}

function entity(identity: EntityIdentity, projectId?: string): LunarEntity {
  return { identity, key: entityKey(identity), projectId, observedAt: 1, authority: 'authoritative', animation: 'work', sourceState: 'running', destination: 'project' }
}

function snapshot(entities: LunarEntity[]): LunarCitySnapshot {
  return { revision: 1, observedAt: 1, sources: [], entities: new Map(entities.map(item => [item.key, item])) }
}

it('groups exact owners across sessions, subagents and Kanban without changing source identities or authority', () => {
  const session = entity({ kind: 'session', connectionId: 'local', profile: 'p', sessionId: 's' }, '/repo')
  const child = entity({ kind: 'subagent', connectionId: 'local', profile: 'p', sessionId: 's', subagentId: 'child' })
  const foreignChild = entity({ kind: 'subagent', connectionId: 'local', profile: 'other', sessionId: 's', subagentId: 'child' })
  const task = entity({ kind: 'kanban', connectionId: 'local', profile: 'p', board: 'b', taskId: 't' }, '/repo')
  const remote = entity({ kind: 'session', connectionId: 'remote', profile: 'p', sessionId: 's' }, '/repo')
  const stale = { ...entity({ kind: 'session', connectionId: 'local', profile: 'p', sessionId: 'old' }, '/repo'), authority: 'stale' as const }
  const input = snapshot([session, child, foreignChild, task, remote, stale])
  const before = JSON.stringify([...input.entities])
  const report = createProjectCompoundController([slot('a', 0), slot('b', 5)]).update(input)
  const local = report.compounds.find(item => item.key === projectCompoundKey('local', '/repo'))!
  expect(local.entityKeys).toEqual([session.key, child.key, task.key, stale.key].sort())
  expect(local.totalCount).toBe(4)
  expect(local.workCount).toBe(3)
  expect(report.compounds).toHaveLength(2)
  expect(report.entityTargets.has(child.key)).toBe(true)
  expect(report.entityTargets.has(stale.key)).toBe(false)
  expect(report.entityTargets.has(foreignChild.key)).toBe(false)
  expect(JSON.stringify([...input.entities])).toBe(before)
})

it('retains occupied slots across order/count changes, rejects overlaps and visibly overflows until a slot frees', () => {
  const controller = createProjectCompoundController([slot('a', 0), slot('duplicate-footprint', 0), slot('b', 5)])
  const make = (projectId: string) => entity({ kind: 'session', connectionId: 'local', profile: 'p', sessionId: projectId }, projectId)
  const z = make('z'), y = make('y'), a = make('a')
  const first = controller.update(snapshot([z, y]))
  const original = new Map(first.compounds.map(item => [item.key, item.slotId]))
  const expanded = controller.update(snapshot([a, y, z]))
  expect(expanded.slotIssues).toHaveLength(1)
  expect(expanded.overflowCount).toBe(1)

  for (const item of expanded.compounds.filter(item => original.has(item.key))) {expect(item.slotId).toBe(original.get(item.key))}
  expect(expanded.entityTargets.has(a.key)).toBe(false)
  const freed = controller.update(snapshot([a, z]))
  expect(freed.overflowCount).toBe(0)
  expect(freed.compounds.find(item => item.projectId === 'z')?.slotId).toBe(original.get(projectCompoundKey('local', 'z')))
  expect(new Set(freed.compounds.map(item => item.slotId)).size).toBe(2)

  const opaque = createProjectCompoundController([slot('only', 0)])
  const canonical = make('/repo'), trailing = make('/repo/'), spaced = make(' /repo ')
  const report = opaque.update(snapshot([canonical, trailing, spaced]))
  expect(report.compounds.map(item => item.projectId).sort()).toEqual([' /repo ', '/repo', '/repo/'])
  expect(new Set(report.compounds.map(item => item.key)).size).toBe(3)
  const overflow = report.compounds.filter(item => item.unplaced)
  expect(overflow).toHaveLength(2)

  for (const group of overflow) {
    expect(group.entityKeys).toEqual([make(group.projectId).key])
    expect(report.entityTargets.has(group.entityKeys[0])).toBe(false)
  }

})
