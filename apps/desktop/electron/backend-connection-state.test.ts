import assert from 'node:assert/strict'
import { type ChildProcess, spawn } from 'node:child_process'
import { once } from 'node:events'

import { test } from 'vitest'

import { waitForBackendExit } from './backend-child'
import { createBackendConnectionState } from './backend-connection-state'

type FakeProcess = { id: string }

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(next => {
    resolve = next
  })

  return { promise, resolve }
}

test('an invalidated remote attempt cannot publish a late descriptor', async () => {
  const state = createBackendConnectionState<FakeProcess, string>()
  const oldProbe = deferred<string>()
  const oldAttempt = state.startAttempt()

  const oldResult = oldProbe.promise.then(descriptor => {
    if (!state.isCurrentAttempt(oldAttempt)) {
      throw new Error('Hermes backend start was superseded by a newer connection attempt.')
    }

    return descriptor
  })

  state.setPromise(oldAttempt, oldResult)
  state.invalidate()

  const newAttempt = state.startAttempt()
  const newResult = Promise.resolve('https://new.example')

  state.setPromise(newAttempt, newResult)
  assert.equal(await newResult, 'https://new.example')

  oldProbe.resolve('https://old.example')
  await assert.rejects(oldResult, /superseded by a newer connection attempt/)
  assert.equal(state.getPromise(), newResult)
})

test('a stale backend exit cannot clear a newer connection attempt', () => {
  const state = createBackendConnectionState<FakeProcess, string>()
  const oldAttempt = state.startAttempt()
  const oldPromise = Promise.resolve('old')

  state.setPromise(oldAttempt, oldPromise)
  const oldOwner = state.attachProcess(oldAttempt, { id: 'old' })
  assert.ok(oldOwner)

  state.invalidate()

  const newAttempt = state.startAttempt()
  const newPromise = Promise.resolve('new')
  const newProcess = { id: 'new' }

  state.setPromise(newAttempt, newPromise)
  assert.ok(state.attachProcess(newAttempt, newProcess))

  assert.equal(state.clearForCurrentProcess(oldOwner), false)
  assert.equal(state.getProcess(), newProcess)
  assert.equal(state.getPromise(), newPromise)
})

test('the current backend exit clears its process and connection promise', () => {
  const state = createBackendConnectionState<FakeProcess, string>()
  const attempt = state.startAttempt()

  state.setPromise(attempt, Promise.resolve('current'))
  const owner = state.attachProcess(attempt, { id: 'current' })
  assert.ok(owner)

  assert.equal(state.clearForCurrentProcess(owner), true)
  assert.equal(state.clearPromiseForAttempt(attempt), true)
  assert.equal(state.getProcess(), null)
  assert.equal(state.getPromise(), null)
})

test('a stale rejected attempt cannot clear a newer connection promise', () => {
  const state = createBackendConnectionState<FakeProcess, string>()
  const oldAttempt = state.startAttempt()

  state.setPromise(oldAttempt, Promise.resolve('old'))
  state.invalidate()

  const newAttempt = state.startAttempt()
  const newPromise = Promise.resolve('new')

  state.setPromise(newAttempt, newPromise)

  assert.equal(state.clearPromiseForAttempt(oldAttempt), false)
  assert.equal(state.getPromise(), newPromise)
})

test('an invalidated attempt cannot attach a late-spawned process', () => {
  const state = createBackendConnectionState<FakeProcess, string>()
  const staleAttempt = state.startAttempt()

  state.invalidate()

  assert.equal(state.attachProcess(staleAttempt, { id: 'late' }), null)
  assert.equal(state.getProcess(), null)
})

test('a failed primary stop retains its child and blocks a replacement until retry exits', async () => {
  const state = createBackendConnectionState<ChildProcess, string>()

  const child = spawn(process.execPath, ['-e', 'process.stdout.write("ready"); setInterval(() => {}, 1000)'], {
    stdio: ['ignore', 'pipe', 'ignore'],
    windowsHide: true
  })

  try {
    await once(child.stdout!, 'data')
    const attempt = state.startAttempt()
    state.setPromise(attempt, Promise.resolve('ready'))
    const owner = state.attachProcess(attempt, child)
    assert.ok(owner)
    const failure = new Error('primary is still running')
    const calls: ChildProcess[] = []

    const fail = async (current: ChildProcess): Promise<void> => {
      calls.push(current)
      throw failure
    }

    const stopping = state.stopProcess(fail)
    assert.equal(state.getProcess(), null)
    assert.equal(state.getPromise(), null)
    assert.throws(() => state.startAttempt(), /has not stopped/)
    assert.equal(state.stopProcess(fail), stopping)
    await assert.rejects(stopping, error => error === failure)
    state.invalidate()
    assert.throws(() => state.startAttempt(), /has not stopped/)
    assert.equal(child.exitCode, null)
    assert.equal(child.signalCode, null)

    await state.stopProcess(async current => {
      calls.push(current)
      current.kill()
      await waitForBackendExit(current, process => {
        process.kill('SIGKILL')
      })
    })
    assert.deepEqual(calls, [child, child])
    assert.ok(child.exitCode !== null || child.signalCode !== null)
    const replacement = state.startAttempt()
    const connection = Promise.resolve('new')
    assert.equal(state.setPromise(replacement, connection), true)
    assert.equal(state.clearForCurrentProcess(owner), false)
    assert.equal(state.getPromise(), connection)
  } finally {
    if (child.exitCode === null && child.signalCode === null) {
      const closed = once(child, 'close')
      child.kill()
      await closed
    }
  }
}, 15_000)

test('shutdown sees a spawned child while its persistent claim is still pending', async () => {
  const state = createBackendConnectionState<ChildProcess, string>()

  const child = spawn(process.execPath, ['-e', 'process.stdout.write("ready"); setInterval(() => {}, 1000)'], {
    stdio: ['ignore', 'pipe', 'ignore'],
    windowsHide: true
  })

  const claim = deferred<void>()

  try {
    await once(child.stdout!, 'data')
    const attempt = state.startAttempt()

    const claiming = state.claimProcess(attempt, child, async current => {
      assert.equal(current, child)
      assert.equal(state.getProcess(), child)
      await claim.promise
    })

    assert.equal(state.getProcess(), child)
    await state.stopProcess(async current => {
      assert.equal(current, child)
      current.kill()
      await waitForBackendExit(current, process => {
        process.kill('SIGKILL')
      })
    })
    assert.ok(child.exitCode !== null || child.signalCode !== null)
    claim.resolve()
    assert.equal(await claiming, null, 'a completed claim cannot revive the stopped generation')
    assert.equal(state.getProcess(), null)
    assert.doesNotThrow(() => state.startAttempt())
  } finally {
    claim.resolve()

    if (child.exitCode === null && child.signalCode === null) {
      const closed = once(child, 'close')
      child.kill()
      await closed
    }
  }
}, 15_000)

test('distinguishes a pending connection attempt from a cached settled descriptor', async () => {
  const state = createBackendConnectionState<FakeProcess, string>()
  const connection = deferred<string>()
  const attempt = state.startAttempt()

  state.setPromise(attempt, connection.promise)
  assert.equal(state.getPendingPromise(), connection.promise)

  connection.resolve('https://remote.example')
  await connection.promise
  await Promise.resolve()

  assert.equal(state.getPromise(), connection.promise)
  assert.equal(state.getPendingPromise(), null)
})
