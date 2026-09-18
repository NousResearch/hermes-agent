import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createPoolRetirer, type PoolRetireEntry } from './pool-retire'
import { LocalBackendSpawnCoordinator } from './pool-spawn-coordinator'

function harness() {
  const coordinator = new LocalBackendSpawnCoordinator(3)
  const pool = new Map<string, PoolRetireEntry>()
  const releases = new Map<string, () => void>()
  const exits = new Map<string, () => void>()
  const stopped: string[] = []
  const cancelled: string[] = []
  const prepared: string[] = []
  let prepare = async (key: string) => `${key}-permit`

  const retirer = createPoolRetirer({
    pool,
    coordinator,
    prepare: async key => {
      prepared.push(key)

      return prepare(key)
    },
    commit: async () => true,
    cancel: async key => { cancelled.push(key) },
    stopBackend: key => new Promise<void>(resolve => {
      stopped.push(key)
      pool.delete(key)
      exits.set(key, () => {
        releases.get(key)?.()
        resolve()
      })
    }),
  })

  async function seed() {
    for (const [index, key] of ['a', 'b', 'c'].entries()) {
      releases.set(key, await coordinator.acquire(key))
      pool.set(key, { process: {}, lastActiveAt: index })
    }
  }

  return { coordinator, pool, releases, exits, stopped, cancelled, prepared, retirer, seed,
    setPrepare: (fn: typeof prepare) => { prepare = fn } }
}

test('reclamation sequences every foreground waiter, including an already queued background promotion', async () => {
  const h = harness()
  await h.seed()
  const first = h.coordinator.request('d', { priority: 'foreground' })
  const second = h.coordinator.request('e', { priority: 'background' })
  second.promote('foreground')
  let granted = 0
  void first.acquired.then(() => { granted += 1 }).catch(() => undefined)
  void second.acquired.then(() => { granted += 1 }).catch(() => undefined)

  try {
    await vi.waitFor(() => assert.deepEqual(h.stopped, ['a']), { timeout: 2000 })
    assert.equal(h.coordinator.activeCount, 3, 'a signal does not release a slot')
    assert.equal(granted, 0)
    assert.throws(() => h.retirer.assertCanOpen('a', 'background'), /retired/i)
    assert.doesNotThrow(() => h.retirer.assertCanOpen('a', 'foreground'))
    h.exits.get('a')!()
    await vi.waitFor(() => assert.deepEqual(h.stopped, ['a', 'b']))
    assert.equal(granted, 1)
    h.exits.get('b')!()
    await Promise.all([first.acquired, second.acquired])
    assert.equal(granted, 2)
    assert.equal(h.coordinator.queuedCount, 0)
  } finally {
    void first.acquired.catch(() => undefined)
    void second.acquired.catch(() => undefined)
    first.cancel()
    second.cancel()
    h.retirer.dispose?.()
  }
})

test('withdrawn demand or replaced candidates cancel prepared authority without stopping a resident', async () => {
  for (const change of ['capacity', 'withdrawal', 'identity'] as const) {
    const h = harness()
    await h.seed()
    let finishPrepare!: (value: string) => void
    h.setPrepare(() => new Promise(resolve => { finishPrepare = resolve }))
    const ticket = h.coordinator.request('d', { priority: 'foreground' })
    void ticket.acquired.catch(() => undefined)

    try {
      await vi.waitFor(() => assert.ok(finishPrepare), { timeout: 2000 })

      if (change === 'capacity') {
        h.releases.get('c')!()
        h.pool.delete('c')
      } else if (change === 'withdrawal') {
        ticket.cancel()
      } else {
        h.pool.set('a', { process: {}, lastActiveAt: 100 })
        ticket.cancel()
      }

      finishPrepare('a-permit')
      await vi.waitFor(() => assert.deepEqual(h.cancelled, ['a']))
      assert.deepEqual(h.stopped, [])
    } finally {
      ticket.cancel()
      h.retirer.dispose?.()
    }
  }
})
