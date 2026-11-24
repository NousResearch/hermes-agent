import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createBootstrapInstallCoordinator } from './bootstrap-install-coordinator'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((r, j) => { resolve = r; reject = j })

  return { promise, resolve, reject }
}

test('concurrent callers join one installation', async () => {
  const coordinator = createBootstrapInstallCoordinator<number>()
  const pending = deferred<number>()
  let starts = 0

  const run = () => { starts++;

 return pending.promise }

  const first = coordinator.run(run)
  const second = coordinator.run(run)
  assert.equal(starts, 1)
  pending.resolve(42)
  assert.equal(await first, 42)
  assert.equal(await second, 42)
  assert.equal(coordinator.cancel(), false)
})

test('cancel waits for the physical installation before another begins', async () => {
  const coordinator = createBootstrapInstallCoordinator<number>()
  const pending = deferred<number>()
  const calls: string[] = []

  const first = coordinator.run(signal => {
    calls.push('first')
    signal.addEventListener('abort', () => calls.push('aborted'))

    return pending.promise
  })

  const cancelling = coordinator.cancelAndWait()

  const second = coordinator.run(async () => { calls.push('second');

 return 2 })

  await Promise.resolve()
  assert.deepEqual(calls, ['first', 'aborted'])
  pending.resolve(1)
  await cancelling
  assert.equal(await first, 1)
  assert.equal(await second, 2)
  assert.deepEqual(calls, ['first', 'aborted', 'second'])
})

test('a waiter invalidated during cancellation cannot start a replacement', async () => {
  const coordinator = createBootstrapInstallCoordinator<number>()
  const pending = deferred<number>()
  const first = coordinator.run(() => pending.promise)
  const cancelling = coordinator.cancelAndWait()
  let valid = true
  let started = false

  const next = coordinator.run(async () => { started = true;

 return 2 }, () => {
    if (!valid) {throw new Error('superseded')}
  })

  valid = false
  pending.resolve(1)
  await first
  await cancelling
  await assert.rejects(next, /superseded/)
  assert.equal(started, false)
})

test('unverified installer termination blocks all replacement installations', async () => {
  const coordinator = createBootstrapInstallCoordinator<number>()
  const pending = deferred<number>()
  const failure = Object.assign(new Error('installer termination unverified'), { installTerminationUnverified: true })
  const first = coordinator.run(() => pending.promise)
  const firstFailure = assert.rejects(first, /termination unverified/)
  const stopping = coordinator.cancelAndWait()
  const stoppingFailure = assert.rejects(stopping, /termination unverified/)
  let starts = 0

  const next = coordinator.run(async () => { starts++;

 return 2 })

  const nextFailure = assert.rejects(next, /termination unverified/)
  pending.reject(failure)
  await Promise.all([firstFailure, stoppingFailure, nextFailure])
  await assert.rejects(coordinator.run(async () => { starts++;

 return 3 }), /termination unverified/)
  assert.equal(starts, 0)
})

test('a non-bootstrap runtime waits for termination and is blocked when termination is unverified', async () => {
  const coordinator = createBootstrapInstallCoordinator<number>()
  const pending = deferred<number>()
  const first = coordinator.run(() => pending.promise)
  const failed = assert.rejects(first, /termination unverified/)
  coordinator.cancel()
  let runtimeStarted = false

  const startExistingRuntime = async () => {
    await coordinator.waitForSafeRuntime()
    runtimeStarted = true
  }

  const starting = startExistingRuntime()
  await Promise.resolve()
  assert.equal(runtimeStarted, false)
  const startingFailed = assert.rejects(starting, /termination unverified/)
  pending.reject(Object.assign(new Error('termination unverified'), { installTerminationUnverified: true }))
  await Promise.all([failed, startingFailed])
  await assert.rejects(startExistingRuntime(), /termination unverified/)
  assert.equal(runtimeStarted, false)
})

test('a non-bootstrap runtime proceeds after verified installation cancellation', async () => {
  const coordinator = createBootstrapInstallCoordinator<number>()
  const pending = deferred<number>()
  const first = coordinator.run(() => pending.promise)
  coordinator.cancel()
  let ready = false
  const waiting = coordinator.waitForSafeRuntime().then(() => { ready = true })
  await Promise.resolve()
  assert.equal(ready, false)
  pending.resolve(1)
  await Promise.all([first, waiting])
  assert.equal(ready, true)
})

test('a rejected cancelled installation cannot clear its replacement', async () => {
  const coordinator = createBootstrapInstallCoordinator<number>()
  const old = deferred<number>()
  const current = deferred<number>()
  const first = coordinator.run(() => old.promise)
  const rejected = assert.rejects(first, /old failure/)
  const stopping = coordinator.cancelAndWait()
  const second = coordinator.run(() => current.promise)
  old.reject(new Error('old failure'))
  await rejected
  await stopping
  assert.equal(coordinator.cancel(), true)
  current.resolve(2)
  assert.equal(await second, 2)
})
