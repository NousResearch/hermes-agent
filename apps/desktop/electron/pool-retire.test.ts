import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createPoolRetirer, type PoolRetireEntry, selectRetirementCandidates } from './pool-retire'
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
          if (outcome === 'replaced') {
            pool.set('a', { process: {}, lastActiveAt: 1 })
          }

          if (outcome === 'fresh') {
            entry.lastActiveAt = Date.now()
          }

          return outcome === 'busy' || outcome === 'unknown' ? null : 'permit'
        },
        commit: async () => {
          events.push('commit')

          return outcome !== 'expired'
        },
        cancel: async key => {
          cancelled.push(key)
        },
        onRetiring: () => {
          events.push('park')
        },
        stopBackend: async key => {
          events.push('stop')
          stopped.push(key)
          pool.delete(key)
        }
      })

      try {
        if (path === 'idle') {
          await retirer.retireIdle('a', 1000)
        } else {
          await retirer.evictTo(0, 1000)
        }

        assert.deepEqual(stopped, outcome === 'idle' ? ['a'] : [], `${path}: ${outcome}`)

        if (outcome === 'idle') {
          assert.deepEqual(events, ['commit', 'park', 'stop'])
        }

        if (['expired', 'replaced', 'fresh'].includes(outcome)) {
          assert.deepEqual(cancelled, ['a'])
        }
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
    ['target', { process: {}, lastActiveAt: 0 }]
  ])

  assert.deepEqual(
    selectRetirementCandidates(pool, new Set(['target'])).map(([key]) => key),
    ['old', 'fresh']
  )
})

test('retireIdle honours the pinned-tier eligibility override while every other safeguard stays armed (#105239)', async () => {
  // The keepalive refreshes lastActiveAt for every open chat, so a pinned
  // backend's legacy clock never expires. The reaper passes an override that
  // also reads lastStreamedAt; the retirer must still veto an active turn,
  // require the admission permit, and keep identity checks — only the idle
  // PREDICATE changes.
  for (const outcome of ['stale-streamed', 'fresh-streamed', 'mid-turn'] as const) {
    const entry: PoolRetireEntry = {
      process: {},
      lastActiveAt: Date.now(),
      // keepalive-fresh in every case; only streamed-activity age varies.
      lastStreamedAt: outcome === 'fresh-streamed' ? Date.now() - 5 * 60_000 : Date.now() - 2 * 60 * 60_000,
      ...(outcome === 'mid-turn' ? { activeTurn: true } : {})
    }
    const pool = new Map([['pinned', entry]])
    const stopped: string[] = []

    const retirer = createPoolRetirer({
      pool,
      coordinator: new LocalBackendSpawnCoordinator(3),
      prepare: async () => 'permit',
      commit: async () => true,
      cancel: async () => {},
      stopBackend: async key => {
        stopped.push(key)
        pool.delete(key)
      }
    })

    try {
      await retirer.retireIdle(
        'pinned',
        10 * 60_000,
        candidate => Boolean(candidate.lastStreamedAt && Date.now() - (candidate.lastStreamedAt || 0) > 60 * 60_000)
      )
    } finally {
      retirer.dispose()
    }

    // stale-streamed (keepalive-fresh, no streamed turn for 2h) retires;
    // fresh-streamed fails the override and stays; mid-turn is vetoed by the
    // retirer even though the override said yes.
    assert.deepEqual(
      stopped,
      outcome === 'stale-streamed' ? ['pinned'] : [],
      outcome
    )
  }
})
