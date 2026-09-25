import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'

import { test } from 'vitest'

import {
  BackgroundSlotRetryBackoff,
  LocalBackendBackgroundCapacityError,
  LocalBackendSlotWaitTimeoutError,
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

test('background slot failures back off per profile and clear after a later success', () => {
  const retries = new BackgroundSlotRetryBackoff({ baseDelayMs: 1_000, maxDelayMs: 8_000 })

  assert.equal(retries.canAttempt('over-cap', 0), true)
  assert.equal(retries.recordFailure('over-cap', 0), 1_000)
  assert.equal(retries.canAttempt('over-cap', 999), false)
  assert.equal(retries.canAttempt('over-cap', 1_000), true)
  assert.equal(retries.recordFailure('over-cap', 1_000), 2_000)
  assert.equal(retries.canAttempt('other-profile', 1_001), true)

  retries.clear('over-cap')
  assert.equal(retries.canAttempt('over-cap', 1_001), true)
})

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

// Spawning 100 native Node processes can exceed Vitest's 5s default on Windows.
// The assertions below still prove concurrency and full cleanup, not wall time.
test('100 real child processes never exceed twelve simultaneous local slots', async () => {
  const limit = 12
  const coordinator = new LocalBackendSpawnCoordinator(limit)
  // Windows may reuse a PID after a child exits. Track child lifetimes and
  // successful exits, rather than requiring 100 globally distinct PID values.
  const liveChildren = new Set<ReturnType<typeof spawn>>()
  let completedChildren = 0
  let maxLive = 0

  await Promise.all(
    Array.from({ length: 100 }, async (_, index) => {
      const release = await coordinator.acquire(`real-profile-${index}`)

      try {
        const child = spawn(process.execPath, ['-e', 'setTimeout(() => {}, 40)'], {
          stdio: 'ignore'
        })

        assert.ok(child.pid)
        liveChildren.add(child)
        maxLive = Math.max(maxLive, liveChildren.size)

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

        completedChildren += 1
        liveChildren.delete(child)
      } finally {
        release()
      }
    })
  )

  // Windows may recycle a PID after a short-lived child exits; completion
  // count, not PID uniqueness, is the invariant this concurrency test owns.
  assert.equal(completedChildren, 100)
  assert.equal(maxLive, limit)
  assert.equal(liveChildren.size, 0)
  assert.equal(coordinator.activeCount, 0)
  assert.equal(coordinator.queuedCount, 0)
}, 30_000)

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

  assert.throws(() => coordinator.request('invalid', { timeoutMs: 0 }), /timeout must be a positive number/)
  assert.throws(() => coordinator.request('invalid', { timeoutMs: Number.NaN }), /timeout must be a positive number/)
  assert.throws(() => coordinator.request('invalid', { timeoutMs: -5 }), /timeout must be a positive number/)

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

test('raising the limit at runtime drains queued waiters into the new slots', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(1)
  const first = await coordinator.acquire('a')
  const queuedB = coordinator.request('b')
  const queuedC = coordinator.request('c')
  await flush()
  assert.equal(coordinator.activeCount, 1)
  assert.equal(coordinator.queuedCount, 2)

  coordinator.setLimit(2)
  const releaseB = await queuedB.acquired
  assert.equal(coordinator.activeCount, 2)
  assert.equal(coordinator.queuedCount, 1)

  first()
  const releaseC = await queuedC.acquired
  assert.equal(coordinator.activeCount, 2)
  releaseB()
  releaseC()
  assert.equal(coordinator.activeCount, 0)
})

test('lowering the limit never revokes granted slots; new requests queue until under cap', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(3)
  const releases = await Promise.all(['a', 'b', 'c'].map(key => coordinator.acquire(key)))
  coordinator.setLimit(1)
  assert.equal(coordinator.activeCount, 3, 'granted slots stay granted')

  const queued = coordinator.request('d')
  await flush()
  assert.equal(coordinator.queuedCount, 1)

  releases[0]()
  releases[1]()
  await flush()
  assert.equal(coordinator.queuedCount, 1, 'still over the new cap of 1')

  releases[2]()
  const releaseD = await queued.acquired
  assert.equal(coordinator.activeCount, 1)
  releaseD()
})

test('setLimit rejects a non-positive or fractional cap', () => {
  const coordinator = new LocalBackendSpawnCoordinator(2)
  assert.throws(() => coordinator.setLimit(0), RangeError)
  assert.throws(() => coordinator.setLimit(1.5), RangeError)
  assert.equal(coordinator.limit, 2)
})

test('cap 3: two background leases leave a reserved slot for foreground', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(3)
  const bg1 = await coordinator.request('bg-1', { priority: 'background' }).acquired
  const bg2 = await coordinator.request('bg-2', { priority: 'background' }).acquired
  assert.equal(coordinator.activeCount, 2)
  assert.equal(coordinator.queuedCount, 0)

  let fgGranted = false

  const fgPromise = coordinator.request('fg', { priority: 'foreground' }).acquired.then(release => {
    fgGranted = true

    return release
  })

  await flush()
  assert.equal(fgGranted, true)
  assert.equal(coordinator.activeCount, 3)

  const releaseFg = await fgPromise
  bg1()
  bg2()
  releaseFg()
  assert.equal(coordinator.activeCount, 0)
})

test('a speculative background request skips saturation without joining the queue', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(3)

  const background = await Promise.all(
    ['bg-1', 'bg-2'].map(key => coordinator.request(key, { priority: 'background' }).acquired)
  )

  const skipped = coordinator.tryRequest('roster-hydration', { priority: 'background' })

  assert.equal(skipped, undefined)
  assert.equal(coordinator.activeCount, 2)
  assert.equal(coordinator.queuedCount, 0)

  const foreground = coordinator.tryRequest('user-click', { priority: 'foreground' })

  assert.ok(foreground)
  assert.equal(foreground.queued, false)
  const releaseForeground = await foreground.acquired

  releaseForeground()

  for (const release of background) {
    release()
  }

  assert.equal(coordinator.activeCount, 0)
})

test('background capacity skips are typed and quiet', () => {
  const error = new LocalBackendBackgroundCapacityError('roster-hydration')

  assert.equal(error.name, 'LocalBackendBackgroundCapacityError')
  assert.equal(error.silent, true)
  assert.match(error.message, /no background slot is currently free/)
})

test('untagged acquire still fills the cap (foreground default)', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(3)
  const releases = await Promise.all(['a', 'b', 'c'].map(key => coordinator.acquire(key)))
  assert.equal(coordinator.activeCount, 3)
  assert.equal(coordinator.queuedCount, 0)

  for (const release of releases) {
    release()
  }

  assert.equal(coordinator.activeCount, 0)
})

test('foreground is granted the reserved slot ahead of a background hydration queue', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(3)

  const bgRunning = await Promise.all(
    ['bg-run-1', 'bg-run-2'].map(key => coordinator.request(key, { priority: 'background' }).acquired)
  )

  const queued = Array.from({ length: 20 }, (_, index) =>
    coordinator.request(`bg-wait-${index}`, { priority: 'background', timeoutMs: 5_000 })
  )

  await flush()
  assert.equal(coordinator.activeCount, 2)
  assert.equal(coordinator.queuedCount, 20)

  const started = Date.now()
  const releaseFg = await coordinator.request('user-click', { priority: 'foreground', timeoutMs: 100 }).acquired
  assert.ok(Date.now() - started < 80, 'foreground must not wait behind the background queue')
  assert.equal(coordinator.activeCount, 3)

  for (const request of queued) {
    request.cancel()
  }

  releaseFg()

  for (const release of bgRunning) {
    release()
  }

  await Promise.all(
    queued.map(request =>
      request.acquired.then(
        () => undefined,
        () => undefined
      )
    )
  )
  assert.equal(coordinator.activeCount, 0)
  assert.equal(coordinator.queuedCount, 0)
})

test('drain prefers a foreground waiter over an earlier background waiter', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(1)
  const releaseHolder = await coordinator.acquire('holder')
  const background = coordinator.request('background', { priority: 'background' })
  const foreground = coordinator.request('foreground', { priority: 'foreground' })
  await flush()
  assert.equal(coordinator.queuedCount, 2)

  let backgroundEntered = false
  let foregroundEntered = false

  const backgroundGrant = background.acquired.then(release => {
    backgroundEntered = true

    return release
  })

  const foregroundGrant = foreground.acquired.then(release => {
    foregroundEntered = true

    return release
  })

  releaseHolder()
  await flush()
  assert.equal(foregroundEntered, true)
  assert.equal(backgroundEntered, false)
  assert.equal(coordinator.activeCount, 1)

  const releaseForeground = await foregroundGrant
  releaseForeground()
  const releaseBackground = await backgroundGrant
  assert.equal(backgroundEntered, true)
  releaseBackground()
  assert.equal(coordinator.activeCount, 0)
})

test('background slot-wait timeout is distinguishable; foreground keeps a user-facing message', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(1)
  const releaseFirst = await coordinator.acquire('first')

  const background = coordinator.request('bg', { priority: 'background', timeoutMs: 10 })
  await assert.rejects(background.acquired, error => {
    assert.ok(error instanceof LocalBackendSlotWaitTimeoutError)
    assert.equal(error.name, 'LocalBackendSlotWaitTimeoutError')
    assert.equal(error.priority, 'background')
    assert.equal(error.silent, true)
    assert.match(error.message, /timed out while waiting for a free slot/)
    assert.match(error.message, /\(background\)/)

    return true
  })

  const foreground = coordinator.request('fg', { priority: 'foreground', timeoutMs: 10 })
  await assert.rejects(foreground.acquired, error => {
    assert.ok(error instanceof Error)
    assert.match(error.message, /timed out while waiting for a free slot/)
    assert.doesNotMatch(error.message, /\(background\)/)
    assert.notEqual(error.name, 'LocalBackendSlotWaitTimeoutError')

    return true
  })

  releaseFirst()
  assert.equal(coordinator.activeCount, 0)
})

test('request() reports whether the caller actually waited behind the queue', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(3)
  const bg1 = coordinator.request('bg-1', { priority: 'background' })
  const bg2 = coordinator.request('bg-2', { priority: 'background' })
  const bgWait = coordinator.request('bg-3', { priority: 'background' })
  assert.equal(bg1.queued, false)
  assert.equal(bg2.queued, false)
  assert.equal(bgWait.queued, true)

  // The reserved slot is free: a foreground request is granted immediately
  // even though a background waiter is queued.
  const fg = coordinator.request('fg', { priority: 'foreground' })
  assert.equal(fg.queued, false)
  assert.equal(coordinator.activeCount, 3)

  bgWait.cancel()
  await bgWait.acquired.catch(() => undefined)
  ;(await fg.acquired)()
  ;(await bg1.acquired)()
  ;(await bg2.acquired)()
  assert.equal(coordinator.activeCount, 0)
})

test('promoting a queued background waiter lets it take the reserved foreground slot', async () => {
  const coordinator = new LocalBackendSpawnCoordinator(3)
  const bg1 = await coordinator.request('bg-1', { priority: 'background' }).acquired
  const bg2 = await coordinator.request('bg-2', { priority: 'background' }).acquired
  const queued = coordinator.request('same-bot', { priority: 'background' })
  await flush()
  assert.equal(coordinator.activeCount, 2)
  assert.equal(coordinator.queuedCount, 1)

  assert.equal(queued.promote('foreground'), true)
  const releasePromoted = await queued.acquired
  assert.equal(coordinator.activeCount, 3)
  assert.equal(coordinator.queuedCount, 0)

  releasePromoted()
  bg1()
  bg2()
  assert.equal(coordinator.activeCount, 0)
})
