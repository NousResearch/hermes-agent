import assert from 'node:assert/strict'

import { test } from 'vitest'

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
