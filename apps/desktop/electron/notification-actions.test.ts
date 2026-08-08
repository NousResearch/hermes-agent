import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolveNotificationAction } from './notification-actions'

const actions = [
  { id: 'approve', text: 'Approve' },
  { id: 'reject', text: 'Reject' }
]

test('resolves an action from Electron 40 event details', () => {
  assert.deepEqual(resolveNotificationAction(actions, { actionIndex: 1 }, undefined), actions[1])
})

test('falls back to the legacy positional action index', () => {
  assert.deepEqual(resolveNotificationAction(actions, {}, 0), actions[0])
})

test('prefers Electron event details over the legacy positional index', () => {
  assert.deepEqual(resolveNotificationAction(actions, { actionIndex: 1 }, 0), actions[1])
})

test('ignores missing and out-of-range action indexes', () => {
  assert.equal(resolveNotificationAction(actions, {}, undefined), null)
  assert.equal(resolveNotificationAction(actions, { actionIndex: -1 }, undefined), null)
  assert.equal(resolveNotificationAction(actions, { actionIndex: 2 }, undefined), null)
})
