import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test, vi } from 'vitest'

import { createPoolStopper, type PoolStopEntry, withPoolEntryAfterStop } from './pool-stop'

// Pins the orphaned-backend class from #62721: fire-and-forget pool stops
// (SIGTERM + immediate entry delete) dropped the process handle before the
// child exited, so a slow/stuck backend survived detached under PID 1.

interface Child {
  exited: boolean
  killed: boolean
}

function harness() {
  const pool = new Map<string, PoolStopEntry>()
  const events: string[] = []
  const exitResolvers = new Map<Child, () => void>()

  const stopper = createPoolStopper({
    pool,
    stopChild: child => {
      ;(child as Child).killed = true
      events.push('stop')
    },
    waitForExit: child =>
      new Promise<void>(resolve => {
        exitResolvers.set(child as Child, () => {
          ;(child as Child).exited = true
          events.push('exit')
          resolve()
        })
      })
  })

  function addChild(key: string): Child {
    const child: Child = { exited: false, killed: false }

    pool.set(key, { process: child })

    return child
  }

  return { addChild, events, exitResolvers, pool, stopper }
}

test('evicts the entry immediately but retains the handle until bounded exit', async () => {
  const { addChild, exitResolvers, pool, stopper } = harness()
  const child = addChild('selena')

  const stop = stopper.stop('selena')

  // No router may hand out the dying backend...
  assert.equal(pool.has('selena'), false)
  // ...but the stop is not done and the in-flight handle is discoverable.
  assert.equal(child.killed, true)
  assert.equal(stopper.inFlight('selena'), stop)

  exitResolvers.get(child)?.()
  await stop

  assert.equal(child.exited, true)
  assert.equal(stopper.inFlight('selena'), undefined)
})

test('concurrent stop requests share one in-flight teardown', async () => {
  const { addChild, events, exitResolvers, stopper } = harness()
  const child = addChild('selena')

  const first = stopper.stop('selena')
  const second = stopper.stop('selena')

  assert.equal(first, second)

  exitResolvers.get(child)?.()
  await first

  // One SIGTERM, one exit — not one per caller.
  assert.deepEqual(events, ['stop', 'exit'])
})

test('stop of an unknown key resolves without signalling anything', async () => {
  const { events, stopper } = harness()

  await stopper.stop('ghost')

  assert.deepEqual(events, [])
  assert.equal(stopper.inFlight('ghost'), undefined)
})

test('a remote pooled descriptor without a local child does not require quit deferral', () => {
  const { pool, stopper } = harness()

  pool.set('remote', {})

  assert.equal(stopper.hasPending(), false)
})

test('stopAll stops every pooled backend and resolves after all exits', async () => {
  const { addChild, exitResolvers, pool, stopper } = harness()
  const a = addChild('a')
  const b = addChild('b')

  let settled = false

  const all = stopper.stopAll().then(() => {
    settled = true
  })

  assert.equal(pool.size, 0)
  assert.equal(a.killed, true)
  assert.equal(b.killed, true)

  exitResolvers.get(a)?.()
  await Promise.resolve()
  assert.equal(settled, false, 'must wait for EVERY child, not the first')

  exitResolvers.get(b)?.()
  await all
  assert.equal(settled, true)
})

test('stopAll joins a stop whose pool entry was already evicted', async () => {
  const { addChild, exitResolvers, pool, stopper } = harness()
  const child = addChild('already-stopping')

  const first = stopper.stop('already-stopping')

  assert.equal(pool.size, 0)
  assert.equal(stopper.hasPending(), true)

  let settled = false

  const all = stopper.stopAll().then(() => {
    settled = true
  })

  await Promise.resolve()

  assert.equal(settled, false)

  exitResolvers.get(child)?.()
  await Promise.all([first, all])
  assert.equal(settled, true)
  assert.equal(stopper.hasPending(), false)
})

test('afterStop holds inFlight until extra teardown finishes (process-less SSH)', async () => {
  const pool = new Map<string, PoolStopEntry>()
  const events: string[] = []
  let releaseAfter: (() => void) | undefined

  const afterGate = new Promise<void>(resolve => {
    releaseAfter = resolve
  })

  const stopper = createPoolStopper({
    pool,
    stopChild: () => {
      events.push('stop')
    },
    waitForExit: async () => {
      events.push('exit')
    },
    afterStop: async () => {
      events.push('after-start')
      await afterGate
      events.push('after-done')
    }
  })

  pool.set('ssh', { process: null })
  const stop = stopper.stop('ssh')
  await Promise.resolve()
  await Promise.resolve()

  assert.equal(stopper.inFlight('ssh'), stop)
  assert.deepEqual(events, ['stop', 'exit', 'after-start'])

  let spawned = false

  const respawn = (async () => {
    const dying = stopper.inFlight('ssh')

    if (dying) {
      await dying
    }

    spawned = true
  })()

  await Promise.resolve()
  assert.equal(spawned, false, 'reconnect must wait for SSH teardown, not just child exit')
  releaseAfter?.()
  await stop
  await respawn
  assert.equal(spawned, true)
  assert.deepEqual(events, ['stop', 'exit', 'after-start', 'after-done'])
  assert.equal(stopper.inFlight('ssh'), undefined)
})

test('a respawn can await the in-flight stop before reusing the key', async () => {
  const { addChild, exitResolvers, stopper } = harness()
  const child = addChild('selena')
  const order: string[] = []

  void stopper.stop('selena')

  const respawn = (async () => {
    const dying = stopper.inFlight('selena')

    if (dying) {
      await dying
    }

    order.push('spawn')
  })()

  order.push('exit-signal')
  exitResolvers.get(child)?.()
  await respawn

  assert.deepEqual(order, ['exit-signal', 'spawn'])
})

test('failed teardown blocks same-profile respawn until the actual late exit', async () => {
  const child = Object.assign(new EventEmitter(), { exitCode: null, signalCode: null })
  const pool = new Map([['profile', { process: child }]])

  const stopper = createPoolStopper({
    pool,
    stopChild: () => {},
    waitForExit: async () => {
      throw new Error('child did not exit')
    }
  })

  const stopping = stopper.stop('profile')

  await assert.rejects(stopping, /did not exit/)
  assert.equal(stopper.inFlight('profile'), stopping)
  assert.equal(stopper.hasPending(), true)
  await assert.rejects(stopper.stop('profile'), /did not exit/)
  child.emit('exit')
  assert.equal(stopper.inFlight('profile'), undefined)
  assert.equal(stopper.hasPending(), false)
})

test('reconnect waits for remote cleanup before looking up or replacing the pooled backend', async () => {
  const key = 'conn:lab::librarian'
  const pool = new Map<string, PoolStopEntry>([[key, { process: null }]])
  const events: string[] = []

  let finishRemoteCleanup = () => {}

  const remoteCleanup = new Promise<void>(resolve => {
    finishRemoteCleanup = resolve
  })

  const stopper = createPoolStopper({
    pool,
    stopChild: () => events.push('stop-local-child'),
    waitForExit: async () => {
      events.push('local-child-exited')
    },
    afterStop: async key => {
      events.push(`remote-cleanup-start:${key}`)
      await remoteCleanup
      events.push(`remote-cleanup-done:${key}`)
    }
  })

  const stop = stopper.stop(key)
  await Promise.resolve()

  const lookup = vi.spyOn(pool, 'get')
  const replacement = { process: null }

  const reconnect = () =>
    withPoolEntryAfterStop(pool, stopper, key, existing => {
      if (existing) {
        return existing
      }

      events.push('reconnect')
      pool.set(key, replacement)

      return replacement
    })

  const first = reconnect()
  const second = reconnect()

  await Promise.resolve()
  assert.equal(lookup.mock.calls.length, 0, 'do not look up the key while remote cleanup is running')
  assert.equal(pool.has(key), false)

  finishRemoteCleanup()
  await stop

  assert.equal(await first, replacement)
  assert.equal(await second, replacement)
  assert.equal(await reconnect(), replacement, 'reuse the backend when no stop is pending')
  assert.deepEqual(events, [
    'stop-local-child',
    'local-child-exited',
    `remote-cleanup-start:${key}`,
    `remote-cleanup-done:${key}`,
    'reconnect'
  ])
})

test('failed teardown prevents reconnect from looking up or replacing the pooled backend', async () => {
  const key = 'conn:lab::librarian'
  const pool = new Map<string, PoolStopEntry>([[key, { process: null }]])
  const failure = new Error('remote cleanup failed')

  const stopper = createPoolStopper({
    pool,
    stopChild: () => {},
    waitForExit: async () => {},
    afterStop: async () => {
      throw failure
    }
  })

  const stop = stopper.stop(key)
  const lookup = vi.spyOn(pool, 'get')
  const reuseOrCreate = vi.fn()
  const reconnect = withPoolEntryAfterStop(pool, stopper, key, reuseOrCreate)

  await Promise.all([
    assert.rejects(stop, error => error === failure),
    assert.rejects(reconnect, error => error === failure)
  ])

  assert.equal(lookup.mock.calls.length, 0)
  assert.equal(reuseOrCreate.mock.calls.length, 0)
})
