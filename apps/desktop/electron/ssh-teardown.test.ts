import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createSshTeardownTracker } from './ssh-teardown'

function deferred() {
  let resolve!: () => void

  const promise = new Promise<void>(done => {
    resolve = done
  })

  return { promise, resolve }
}

test('quit joins teardown whose SSH routing entry was already removed', async () => {
  const exit = deferred()
  const tracker = createSshTeardownTracker()
  const close = vi.fn(async () => {})
  const stopped = tracker.track({ close }, () => exit.promise)
  let done = false

  const quit = tracker
    .finish([], async () => {})
    .then(() => {
      done = true
    })

  await Promise.resolve()
  assert.equal(done, false)
  assert.equal(tracker.hasPending(), true)
  exit.resolve()
  await Promise.all([stopped, quit])
  assert.equal(tracker.hasPending(), false)
})

test('a stuck SSH teardown and force cleanup cannot hang quit forever', async () => {
  vi.useFakeTimers()

  try {
    const tracker = createSshTeardownTracker()
    const close = vi.fn(() => new Promise<void>(() => {}))
    void tracker.track({ close }, () => new Promise<void>(() => {}))
    const force = vi.fn(() => new Promise<void>(() => {}))
    const quit = tracker.finish([], force, 20, 10)
    await vi.advanceTimersByTimeAsync(30)
    await quit
    assert.equal(close.mock.calls.length, 1)
    assert.equal(force.mock.calls.length, 1)
    assert.equal(vi.getTimerCount(), 0)
  } finally {
    vi.useRealTimers()
  }
})
