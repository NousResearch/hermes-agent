import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createBackendConnectionState } from './backend-connection-state'
import { resolveProfileApiRequest } from './connection-config'

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

test('remembering another startup profile cannot retarget config reads or saves on a live backend', () => {
  const state = createBackendConnectionState<FakeProcess, string>()
  let rememberedProfile = 'default'
  const attempt = state.startAttempt(rememberedProfile)
  state.setPromise(attempt, Promise.resolve('default-backend'))
  rememberedProfile = 'writer'

  for (const method of ['GET', 'PUT']) {
    assert.deepEqual(resolveProfileApiRequest('writer', '/api/config', {
      primaryProfile: state.getProfile() || rememberedProfile,
      requestMethod: method,
    }), { backendProfile: null, requestPath: '/api/config?profile=writer' })
  }

  state.invalidate()
  state.startAttempt(rememberedProfile)
  assert.equal(state.getProfile(), 'writer')
  assert.deepEqual(resolveProfileApiRequest('writer', '/api/config', {
    primaryProfile: state.getProfile() || rememberedProfile,
    requestMethod: 'PUT',
  }), { backendProfile: null, requestPath: '/api/config' })
})

test('only the current backend lifecycle can clear the captured primary profile', () => {
  const state = createBackendConnectionState<FakeProcess, string>()
  const oldAttempt = state.startAttempt('default')
  state.setPromise(oldAttempt, Promise.resolve('old'))
  const oldOwner = state.attachProcess(oldAttempt, { id: 'old' })!
  state.invalidate()
  assert.equal(state.getProfile(), null)

  const current = state.startAttempt('writer')
  state.setPromise(current, Promise.resolve('current'))
  const owner = state.attachProcess(current, { id: 'current' })!
  assert.equal(state.clearForCurrentProcess(oldOwner), false)
  assert.equal(state.clearPromiseForAttempt(oldAttempt), false)
  assert.equal(state.getProfile(), 'writer')
  assert.equal(state.clearForCurrentProcess(owner), true)
  assert.equal(state.getProfile(), null)

  // Remote attempts have a promise but no child process: rejection must also
  // release the captured identity so a real reconnect can choose a new one.
  const remote = state.startAttempt('remote-profile')
  state.setPromise(remote, Promise.resolve('remote'))
  assert.equal(state.getProfile(), 'remote-profile')
  assert.equal(state.clearPromiseForAttempt(remote), true)
  assert.equal(state.getProfile(), null)
})
