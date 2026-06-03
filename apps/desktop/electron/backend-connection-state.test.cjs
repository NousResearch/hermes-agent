const assert = require('node:assert/strict')
const test = require('node:test')

const { createBackendConnectionState } = require('./backend-connection-state.cjs')

test('stale local process exit cannot clear a newer connection promise', () => {
  const state = createBackendConnectionState()
  const staleProcess = { killed: false }
  const staleAttempt = state.startAttempt()
  const stalePromise = Promise.resolve('local')

  state.setPromise(staleAttempt, stalePromise)
  const staleOwner = state.attachProcess(staleProcess)

  state.invalidate()

  const remoteAttempt = state.startAttempt()
  const remotePromise = Promise.resolve('remote')
  state.setPromise(remoteAttempt, remotePromise)

  assert.equal(state.clearForCurrentProcess(staleOwner), false)
  assert.equal(state.getPromise(), remotePromise)
})

test('current local process exit clears the active connection promise', () => {
  const state = createBackendConnectionState()
  const process = { killed: false }
  const attempt = state.startAttempt()
  const promise = Promise.resolve('local')

  state.setPromise(attempt, promise)
  const owner = state.attachProcess(process)

  assert.equal(state.clearForCurrentProcess(owner), true)
  assert.equal(state.getProcess(), null)
  assert.equal(state.getPromise(), null)
})

test('stale rejected attempt cannot clear a newer connection promise', () => {
  const state = createBackendConnectionState()
  const staleAttempt = state.startAttempt()
  const stalePromise = Promise.resolve('local')

  state.setPromise(staleAttempt, stalePromise)
  state.invalidate()

  const remoteAttempt = state.startAttempt()
  const remotePromise = Promise.resolve('remote')
  state.setPromise(remoteAttempt, remotePromise)

  assert.equal(state.clearPromiseForAttempt(staleAttempt), false)
  assert.equal(state.getPromise(), remotePromise)
})
