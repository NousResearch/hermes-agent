import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  poolKeyProfile,
  runningAssigneesFromKanbanList,
  shouldKeepPoolBackendWarm
} from './pool-kanban-warm'

test('poolKeyProfile unwraps connection-scoped keys', () => {
  assert.equal(poolKeyProfile('architect'), 'architect')
  assert.equal(poolKeyProfile('conn:office::engineer'), 'engineer')
  assert.equal(poolKeyProfile('default'), 'default')
})

test('runningAssigneesFromKanbanList keeps only running assignees', () => {
  const running = runningAssigneesFromKanbanList([
    { id: 't_a', status: 'running', assignee: 'Architect' },
    { id: 't_b', status: 'ready', assignee: 'engineer' },
    { id: 't_c', status: 'running', assignee: 'qa-verifier' },
    { id: 't_d', status: 'todo', assignee: 'reviewer' }
  ])
  assert.deepEqual([...running].sort(), ['architect', 'qa-verifier'])
})

test('shouldKeepPoolBackendWarm skips idle kill for a running seat', () => {
  const running = new Set(['architect', 'engineer'])
  assert.equal(shouldKeepPoolBackendWarm('architect', running), true)
  assert.equal(shouldKeepPoolBackendWarm('conn:studio::engineer', running), true)
  assert.equal(shouldKeepPoolBackendWarm('reviewer', running), false)
})

test('empty or malformed kanban payloads keep no one warm', () => {
  assert.equal(runningAssigneesFromKanbanList(null).size, 0)
  assert.equal(runningAssigneesFromKanbanList({}).size, 0)
  assert.equal(shouldKeepPoolBackendWarm('architect', new Set()), false)
})
