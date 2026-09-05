// Provenance: carried forward from worker commit
// 475d6819e58da4b18c32a6f34cf55e5e4d0f26df on
// codex/hermes-remote-session-fanout-proof-192035-20260904. The parent is the
// immutable production source pin 2ddd96aff552b7aa8c48a5543b1687a41cf26c02.
// The parent manager independently ran both cases against that source (2/2
// passed). This is test-only; no worker production changes or ancestry are
// imported.

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { fetchRegistrySessionRows, fetchRemoteProfileSessions } from './profile-session-routing'

test('remote registry fanout preserves valid rows when a sibling gateway times out', async () => {
  const calls: Array<{ descriptor: unknown; path: string }> = []
  let markTimeoutStarted!: () => void
  let releaseTimeout!: () => void
  const timeoutStarted = new Promise<void>(resolve => {
    markTimeoutStarted = resolve
  })
  const timeoutRelease = new Promise<void>(resolve => {
    releaseTimeout = resolve
  })
  let liveStarted = false

  const rowsPromise = fetchRegistrySessionRows(
    [
      { connectionId: 'gw-timeout', kind: 'ssh', backends: [{ descriptor: 'timeout', profileLabel: 'slow' }] },
      { connectionId: 'gw-live', kind: 'ssh', backends: [{ descriptor: 'live', profileLabel: 'fast' }] }
    ],
    new URLSearchParams({ profile: 'all', limit: '20', offset: '0' }),
    async (descriptor, path) => {
      calls.push({ descriptor, path })

      if (descriptor === 'timeout') {
        markTimeoutStarted()
        await timeoutRelease
        throw new Error('ETIMEDOUT')
      }

      liveStarted = true
      return { sessions: [{ id: 'live-session' }], total: 1 }
    }
  )

  await timeoutStarted
  const fanoutOverlapped = liveStarted
  releaseTimeout()
  const rows = await rowsPromise

  assert.deepEqual(
    rows.map(row => ({
      id: (row as { id: string }).id,
      connection_id: (row as { connection_id: string }).connection_id,
      profile: (row as { profile: string }).profile
    })),
    [{ id: 'live-session', connection_id: 'gw-live', profile: 'fast' }]
  )
  assert.equal(fanoutOverlapped, true, 'live sibling must start before the timeout is released')
  assert.deepEqual(
    calls.map(call => call.descriptor),
    ['timeout', 'live']
  )
  assert.ok(calls.every(call => call.path === '/api/sessions?limit=20&offset=0'))
})

test('remote profile timeout rejects instead of becoming an empty successful response', async () => {
  const timeout = new Error('ETIMEDOUT')

  await assert.rejects(
    fetchRemoteProfileSessions(
      'remote-timeout',
      new URLSearchParams({ profile: 'remote-timeout', limit: '20', offset: '0' }),
      async () => {
        await new Promise(resolve => setTimeout(resolve, 5))
        throw timeout
      }
    ),
    error => error === timeout
  )
})
