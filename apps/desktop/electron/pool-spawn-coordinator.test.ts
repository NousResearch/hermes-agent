import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'

import { test } from 'vitest'

import {
  LocalBackendSpawnCoordinator,
  releaseLocalBackendSlotAfterExit
} from './pool-spawn-coordinator'

const deferred = () => {
  let resolve!: () => void
  const promise = new Promise<void>(done => {
    resolve = done
  })
  return { promise, resolve }
}

const flush = () => new Promise<void>(resolve => setImmediate(resolve))

test('100 concurrent local requests never hold more than the configured slots', async () => {
  const limit = 12
  const coordinator = new LocalBackendSpawnCoordinator(limit)
  const gates = Array.from({ length: 100 }, deferred)
  let active = 0
  let maxActive = 0

  const tasks = gates.map(async (gate, index) => {
    const release = await coordinator.acquire(`profile-${index}`)
    active += 1
    maxActive = Math.max(maxActive, active)

    await gate.promise

    active -= 1
    release()
  })

  await flush()
  assert.equal(active, limit)
  assert.equal(coordinator.activeCount, limit)
  assert.equal(coordinator.queuedCount, 100 - limit)

  for (let start = 0; start < gates.length; start += limit) {
    for (const gate of gates.slice(start, start + limit)) {
      gate.resolve()
    }
    await flush()
  }

  await Promise.all(tasks)
  assert.equal(maxActive, limit)
  assert.equal(coordinator.activeCount, 0)
  assert.equal(coordinator.queuedCount, 0)
})

test('a queued start can be cancelled without waiting for an active backend', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(1)
  const releaseFirst = await coordinator.acquire('first')
  const queued = coordinator.request('cancelled')

  assert.equal(coordinator.queuedCount, 1)
  assert.equal(queued.cancel(), true)
  await assert.rejects(queued.acquired, /cancelled while queued/)
  assert.equal(coordinator.activeCount, 1)
  assert.equal(coordinator.queuedCount, 0)

  releaseFirst()
  assert.equal(coordinator.activeCount, 0)
})

test('cancelling an old same-key request never rejects a newer waiter', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(1)
  const blocker = coordinator.request('blocker')
  const releaseBlocker = await blocker.acquired
  const old = coordinator.request('same-profile')

  releaseBlocker()
  const newer = coordinator.request('same-profile')

  assert.equal(old.cancel(), false, 'the old request was already granted')
  assert.equal(coordinator.queuedCount, 1, 'the newer same-key waiter must remain queued')

  const releaseOld = await old.acquired
  releaseOld()
  const releaseNewer = await newer.acquired
  releaseNewer()

  assert.equal(coordinator.activeCount, 0)
  assert.equal(coordinator.queuedCount, 0)
})

test('a queued start times out with a clear error and frees its queue position', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(1)
  const releaseFirst = await coordinator.acquire('first')
  const queued = coordinator.request('timed-out', { timeoutMs: 10 })

  await assert.rejects(queued.acquired, /timed out while waiting for a free slot/)
  assert.equal(coordinator.activeCount, 1)
  assert.equal(coordinator.queuedCount, 0)

  releaseFirst()
  assert.equal(coordinator.activeCount, 0)
})

test('100 real child processes never exceed twelve simultaneous local slots', async () => {
  const limit = 12
  const coordinator = new LocalBackendSpawnCoordinator(limit)
  const livePids = new Set<number>()
  const seenPids = new Set<number>()
  let maxLive = 0

  await Promise.all(
    Array.from({ length: 100 }, async (_, index) => {
      const release = await coordinator.acquire(`real-profile-${index}`)

      try {
        const child = spawn(process.execPath, ['-e', 'setTimeout(() => {}, 40)'], {
          stdio: 'ignore'
        })
        assert.ok(child.pid)
        livePids.add(child.pid)
        seenPids.add(child.pid)
        maxLive = Math.max(maxLive, livePids.size)

        await new Promise<void>((resolve, reject) => {
          child.once('error', reject)
          child.once('exit', code => {
            if (code === 0) {
              resolve()
            } else {
              reject(new Error(`child ${child.pid} exited with ${code}`))
            }
          })
        })

        livePids.delete(child.pid)
      } finally {
        release()
      }
    })
  )

  assert.equal(seenPids.size, 100)
  assert.equal(maxLive, limit)
  assert.equal(livePids.size, 0)
  assert.equal(coordinator.activeCount, 0)
  assert.equal(coordinator.queuedCount, 0)
})

test('failed start keeps its slot until the child has actually exited', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(1)
  const childExit = deferred()
  const releaseFailed = await coordinator.acquire('failed')
  let successorEntered = false
  const successor = coordinator.acquire('successor').then(release => {
    successorEntered = true
    return release
  })

  const cleanup = releaseLocalBackendSlotAfterExit(releaseFailed, () => childExit.promise)
  await flush()

  assert.equal(successorEntered, false)
  assert.equal(coordinator.activeCount, 1)
  assert.equal(coordinator.queuedCount, 1)

  childExit.resolve()
  await cleanup
  const releaseSuccessor = await successor

  assert.equal(successorEntered, true)
  assert.equal(coordinator.activeCount, 1)
  assert.equal(coordinator.queuedCount, 0)

  releaseSuccessor()
  assert.equal(coordinator.activeCount, 0)
})

test('a rejected wait keeps the slot occupied', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(1)
  const releaseFailed = await coordinator.acquire('failed')
  let successorEntered = false
  const successor = coordinator.acquire('successor').then(release => {
    successorEntered = true
    return release
  })

  const cleanup = releaseLocalBackendSlotAfterExit(releaseFailed, async () => {
    throw new Error('exit unproven')
  })

  await assert.rejects(cleanup, /exit unproven/)
  await flush()

  assert.equal(successorEntered, false)
  assert.equal(coordinator.activeCount, 1)
  assert.equal(coordinator.queuedCount, 1)

  releaseFailed()
  const releaseSuccessor = await successor
  assert.equal(successorEntered, true)
  releaseSuccessor()
  assert.equal(coordinator.activeCount, 0)
  assert.equal(coordinator.queuedCount, 0)
})

test('an invalid timeout never enqueues a waiter', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(1)
  const releaseFirst = await coordinator.acquire('first')

  assert.throws(
    () => coordinator.request('invalid', { timeoutMs: 0 }),
    /timeout must be a positive number/
  )
  assert.throws(
    () => coordinator.request('invalid', { timeoutMs: Number.NaN }),
    /timeout must be a positive number/
  )
  assert.throws(
    () => coordinator.request('invalid', { timeoutMs: -5 }),
    /timeout must be a positive number/
  )

  assert.equal(coordinator.activeCount, 1)
  assert.equal(coordinator.queuedCount, 0)

  releaseFirst()
  assert.equal(coordinator.activeCount, 0)
})

test('a failed or repeated cleanup releases exactly one slot', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(1)
  const releaseFirst = await coordinator.acquire('first')
  let secondEntered = false
  const second = coordinator.acquire('second').then(release => {
    secondEntered = true
    return release
  })

  await flush()
  assert.equal(secondEntered, false)
  assert.equal(coordinator.activeCount, 1)
  assert.equal(coordinator.queuedCount, 1)

  releaseFirst()
  releaseFirst()
  const releaseSecond = await second

  assert.equal(secondEntered, true)
  assert.equal(coordinator.activeCount, 1)
  assert.equal(coordinator.queuedCount, 0)

  releaseSecond()
  assert.equal(coordinator.activeCount, 0)
})
