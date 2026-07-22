import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  createProfileBackendStartupQueue,
  normalizeProfileBackendStartReason,
  reusePoolConnection
} from './profile-backend-startup'

test('serializes concurrent local profile cold starts', async () => {
  const queue = createProfileBackendStartupQueue()
  let active = 0
  let maximumActive = 0
  let releaseFirst: () => void

  const firstGate = new Promise<void>(resolve => {
    releaseFirst = resolve
  })

  let markFirstStarted: () => void

  const firstStarted = new Promise<void>(resolve => {
    markFirstStarted = resolve
  })

  const started: string[] = []

  const first = queue.run(async () => {
    active += 1
    maximumActive = Math.max(maximumActive, active)
    started.push('first')
    markFirstStarted!()
    await firstGate
    active -= 1

    return 'first'
  })

  const second = queue.run(async () => {
    active += 1
    maximumActive = Math.max(maximumActive, active)
    started.push('second')
    active -= 1

    return 'second'
  })

  await firstStarted
  assert.deepEqual(started, ['first'])
  releaseFirst!()
  assert.deepEqual(await Promise.all([first, second]), ['first', 'second'])
  assert.deepEqual(started, ['first', 'second'])
  assert.equal(maximumActive, 1)
})

test('a failed local startup releases the next queued profile', async () => {
  const queue = createProfileBackendStartupQueue()

  const first = queue.run(async () => {
    throw new Error('profile failed to start')
  })

  const second = queue.run(async () => 'ready')

  await assert.rejects(first, /profile failed to start/)
  assert.equal(await second, 'ready')
})

test('reuses an existing backend promise immediately while another profile starts', async () => {
  const connection = Promise.resolve({ profile: 'already-running' })
  const entry = { connectionPromise: connection, lastActiveAt: 1 }

  assert.equal(reusePoolConnection(entry, 99), connection)
  assert.equal(entry.lastActiveAt, 99)
})

test('normalizes untrusted renderer start reasons to unknown', () => {
  assert.equal(normalizeProfileBackendStartReason('profile_activate'), 'profile_activate')
  assert.equal(normalizeProfileBackendStartReason('not-a-reason'), 'unknown')
  assert.equal(normalizeProfileBackendStartReason({}), 'unknown')
})
