import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createPoolRetirer, teardownStopsBackend, type PoolRetireEntry, selectRetirementCandidates } from './pool-retire'
import { LocalBackendSpawnCoordinator } from './pool-spawn-coordinator'

test('idle and LRU retirement require backend authority, unchanged identity and current eligibility', async () => {
  for (const path of ['idle', 'lru'] as const) {
    for (const outcome of ['busy', 'unknown', 'expired', 'replaced', 'fresh', 'idle'] as const) {
      const entry: PoolRetireEntry = { process: {}, lastActiveAt: 1 }
      const pool = new Map([['a', entry]])
      const cancelled: string[] = []
      const stopped: string[] = []
      const events: string[] = []

      const retirer = createPoolRetirer({
        pool,
        coordinator: new LocalBackendSpawnCoordinator(3),
        prepare: async () => {
          if (outcome === 'replaced') {pool.set('a', { process: {}, lastActiveAt: 1 })}

          if (outcome === 'fresh') {entry.lastActiveAt = Date.now()}

          return outcome === 'busy' || outcome === 'unknown' ? null : 'permit'
        },
        commit: async () => {
          events.push('commit')

          return outcome !== 'expired'
        },
        cancel: async key => { cancelled.push(key) },
        onRetiring: () => { events.push('park') },
        stopBackend: async key => { events.push('stop'); stopped.push(key); pool.delete(key) },
      })

      try {
        if (path === 'idle') {await retirer.retireIdle('a', 1000)}
        else {await retirer.evictTo(0, 1000)}

        assert.deepEqual(stopped, outcome === 'idle' ? ['a'] : [], `${path}: ${outcome}`)

        if (outcome === 'idle') {assert.deepEqual(events, ['commit', 'park', 'stop'])}

        if (['expired', 'replaced', 'fresh'].includes(outcome)) {assert.deepEqual(cancelled, ['a'])}
      } finally {
        retirer.dispose()
      }
    }
  }
})

test('candidate selection excludes processless descriptors, renderer-leased work and queued target scopes', () => {
  const pool = new Map<string, PoolRetireEntry>([
    ['fresh', { process: {}, lastActiveAt: 100 }],
    ['old', { process: {}, lastActiveAt: 1 }],
    ['busy', { process: {}, lastActiveAt: 0, activeTurn: true }],
    ['descriptor', { process: null }],
    ['target', { process: {}, lastActiveAt: 0 }],
  ])

  assert.deepEqual(selectRetirementCandidates(pool, new Set(['target'])).map(([key]) => key), ['old', 'fresh'])
})

/** A pooled descriptor (registry/remote) has no child, but a live backend behind it. */
test('a stale pooled backend whose teardown kills it is reclaimed only through its work proof', async () => {
  for (const outcome of ['busy', 'unknown', 'idle'] as const) {
    const entry: PoolRetireEntry = {
      process: null,
      connectionPromise: Promise.resolve({ baseUrl: 'http://127.0.0.1:9', mode: 'remote' }),
      lastActiveAt: 1
    }
    const pool = new Map([['remote', entry]])
    const prepared: string[] = []
    const stopped: string[] = []
    const retirer = createPoolRetirer({
      pool,
      coordinator: new LocalBackendSpawnCoordinator(3),
      prepare: async key => {
        prepared.push(key)

        return outcome === 'idle' ? 'permit' : null
      },
      commit: async () => true,
      cancel: async () => undefined,
      stopBackend: async key => { stopped.push(key); pool.delete(key) },
    })

    try {
      // The reaper only routes a descriptor here when its teardown really stops
      // the backend (an SSH `serve` this app spawned) — main.ts:startPoolIdleReaper.
      await retirer.retireIdle('remote', 1000)

      // The backend-side proof is the ONLY thing that authorizes a teardown, so
      // it must be consulted for a descriptor entry too — a remote backend can
      // be mid-cron just as a local child can.
      assert.deepEqual(prepared, ['remote'], `proof consulted: ${outcome}`)
      assert.deepEqual(stopped, outcome === 'idle' ? ['remote'] : [], `teardown: ${outcome}`)
    } finally {
      retirer.dispose()
    }
  }
})

test('only a teardown that stops a backend process demands the work proof', () => {
  assert.equal(teardownStopsBackend({ process: {} }, false), true)
  // A remote `serve` this app spawned over SSH: its teardown kills the child.
  assert.equal(teardownStopsBackend({ process: null, connectionPromise: Promise.resolve({}) }, true), true)
  // A plain remote descriptor is a local handle: dropping it kills nothing, and
  // the fence behind it is irreversible.
  assert.equal(teardownStopsBackend({ process: null, connectionPromise: Promise.resolve({}) }, false), false)
  assert.equal(teardownStopsBackend({ process: null, connectionPromise: null }, false), false)
  // Ownership alone decides: the entry shape is not what gates the proof.
  assert.equal(teardownStopsBackend({}, true), true)
})
