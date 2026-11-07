import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  releaseBridgeOwnerAndStopSidecarIfIdle,
  scheduleBridgeReconnectIfCurrent,
  ScopedComputerUseBridgeLifecycle
} from './computer-use-bridge-lifecycle'

test('sidecar startup timeout and early exit schedule retry for a current enabled owner', async () => {
  const lifecycle = new ScopedComputerUseBridgeLifecycle<any>()
  const remoteKey = 'https://remote.example|token|profile|current'
  const generation = lifecycle.generation(remoteKey)
  lifecycle.acquire(remoteKey, 'pool:research')

  const failures = [
    new Error('Timed out waiting for local Computer Use bridge to announce its port.'),
    new Error('Local Computer Use bridge exited before ready (1).')
  ]

  let reconnects = 0

  for (const failure of failures) {
    await assert.rejects(
      Promise.reject(failure).catch(error => {
        const scheduled = scheduleBridgeReconnectIfCurrent({
          lifecycle,
          remoteKey,
          stopping: false,
          enabled: true,
          capturedGlobalGeneration: 4,
          currentGlobalGeneration: 4,
          capturedScopedGeneration: generation,
          scheduleReconnect: () => (reconnects += 1)
        })

        assert.equal(scheduled, true)
        throw error
      }),
      error => error === failure
    )
  }

  assert.equal(reconnects, 2)
})

test('sidecar startup failure does not retry a stale, disabled, stopping, or unowned scope', () => {
  const remoteKey = 'https://remote.example|token|profile|current'

  const cases = [
    { name: 'stale global generation', stopping: false, enabled: true, capturedGlobal: 3, currentGlobal: 4 },
    { name: 'disabled bridge', stopping: false, enabled: false, capturedGlobal: 4, currentGlobal: 4 },
    { name: 'stopping bridge', stopping: true, enabled: true, capturedGlobal: 4, currentGlobal: 4 }
  ]

  for (const testCase of cases) {
    const lifecycle = new ScopedComputerUseBridgeLifecycle<any>()
    lifecycle.acquire(remoteKey, 'pool:research')
    const scopedGeneration = lifecycle.generation(remoteKey)
    let reconnects = 0

    const scheduled = scheduleBridgeReconnectIfCurrent({
      lifecycle,
      remoteKey,
      stopping: testCase.stopping,
      enabled: testCase.enabled,
      capturedGlobalGeneration: testCase.capturedGlobal,
      currentGlobalGeneration: testCase.currentGlobal,
      capturedScopedGeneration: scopedGeneration,
      scheduleReconnect: () => (reconnects += 1)
    })

    assert.equal(scheduled, false, testCase.name)
    assert.equal(reconnects, 0, testCase.name)
  }

  const lifecycle = new ScopedComputerUseBridgeLifecycle<any>()
  lifecycle.acquire(remoteKey, 'pool:research')
  const staleScopedGeneration = lifecycle.generation(remoteKey)
  lifecycle.cancel(remoteKey)
  let reconnects = 0

  assert.equal(
    scheduleBridgeReconnectIfCurrent({
      lifecycle,
      remoteKey,
      stopping: false,
      enabled: true,
      capturedGlobalGeneration: 4,
      currentGlobalGeneration: 4,
      capturedScopedGeneration: staleScopedGeneration,
      scheduleReconnect: () => (reconnects += 1)
    }),
    false,
    'stale scoped generation'
  )
  assert.equal(reconnects, 0)

  const currentGeneration = lifecycle.generation(remoteKey)
  lifecycle.release(remoteKey, 'pool:research')
  assert.equal(
    scheduleBridgeReconnectIfCurrent({
      lifecycle,
      remoteKey,
      stopping: false,
      enabled: true,
      capturedGlobalGeneration: 4,
      currentGlobalGeneration: 4,
      capturedScopedGeneration: currentGeneration,
      scheduleReconnect: () => (reconnects += 1)
    }),
    false,
    'unowned scope'
  )
  assert.equal(reconnects, 0)
})

test('releasing one scoped bridge preserves its sibling connection and the shared sidecar', () => {
  const cleared: unknown[] = []
  const lifecycle = new ScopedComputerUseBridgeLifecycle<any>(timer => cleared.push(timer))
  let aliceCloses = 0
  let bobCloses = 0
  const alice = { closedByDesktop: false, ws: { close: () => (aliceCloses += 1) } }
  const bob = { closedByDesktop: false, ws: { close: () => (bobCloses += 1) } }
  const aliceTimer = { owner: 'alice' } as unknown as ReturnType<typeof setTimeout>
  const bobTimer = { owner: 'bob' } as unknown as ReturnType<typeof setTimeout>
  const alicePending = Promise.resolve(alice)
  const bobPending = Promise.resolve(bob)
  let sidecarStops = 0

  lifecycle.acquire('alice', 'pool:alice')
  lifecycle.acquire('bob', 'pool:bob')
  lifecycle.connections.set('alice', alice)
  lifecycle.connections.set('bob', bob)
  lifecycle.connectionPromises.set('alice', alicePending)
  lifecycle.connectionPromises.set('bob', bobPending)
  lifecycle.reconnectTimers.set('alice', aliceTimer)
  lifecycle.reconnectTimers.set('bob', bobTimer)
  const aliceGeneration = lifecycle.generation('alice')
  const bobGeneration = lifecycle.generation('bob')

  const stoppedSidecar = releaseBridgeOwnerAndStopSidecarIfIdle({
    lifecycle,
    remoteKey: 'alice',
    owner: 'pool:alice',
    stopSidecar: () => (sidecarStops += 1)
  })

  assert.equal(stoppedSidecar, false)
  assert.equal(sidecarStops, 0)
  assert.equal(aliceCloses, 1)
  assert.equal(alice.closedByDesktop, true)
  assert.equal(lifecycle.connections.has('alice'), false)
  assert.equal(lifecycle.connectionPromises.has('alice'), false)
  assert.equal(lifecycle.reconnectTimers.has('alice'), false)
  assert.equal(lifecycle.hasOwners('alice'), false)
  assert.deepEqual(cleared, [aliceTimer])
  assert.equal(lifecycle.isCurrent('alice', aliceGeneration), false)

  assert.equal(bobCloses, 0)
  assert.equal(bob.closedByDesktop, false)
  assert.equal(lifecycle.connections.get('bob'), bob)
  assert.equal(lifecycle.connectionPromises.get('bob'), bobPending)
  assert.equal(lifecycle.reconnectTimers.get('bob'), bobTimer)
  assert.equal(lifecycle.hasOwners('bob'), true)
  assert.equal(lifecycle.isCurrent('bob', bobGeneration), true)
})

test('releasing the final bridge owner stops and clears the local sidecar without reconnecting', () => {
  const cleared: unknown[] = []
  const lifecycle = new ScopedComputerUseBridgeLifecycle<any>(timer => cleared.push(timer))
  const remoteKey = 'https://remote.example|token|profile|current'
  const owner = 'primary'
  let closes = 0
  let stops = 0
  const state = { closedByDesktop: false, ws: { close: () => (closes += 1) } }
  const timer = { reconnect: remoteKey } as unknown as ReturnType<typeof setTimeout>

  const sidecar = {
    process: { killed: false } as { killed: boolean } | null,
    startPromise: Promise.resolve() as Promise<void> | null,
    state: { token: 'secret', url: 'http://127.0.0.1:1234' } as object | null
  }

  lifecycle.acquire(remoteKey, owner)
  lifecycle.connections.set(remoteKey, state)
  lifecycle.connectionPromises.set(remoteKey, Promise.resolve(state))
  lifecycle.reconnectTimers.set(remoteKey, timer)
  const generation = lifecycle.generation(remoteKey)

  const stopped = releaseBridgeOwnerAndStopSidecarIfIdle({
    lifecycle,
    remoteKey,
    owner,
    stopSidecar: () => {
      stops += 1

      if (sidecar.process) {
        sidecar.process.killed = true
      }

      sidecar.process = null
      sidecar.startPromise = null
      sidecar.state = null
    }
  })

  assert.equal(stopped, true)
  assert.equal(stops, 1)
  assert.equal(sidecar.process, null)
  assert.equal(sidecar.startPromise, null)
  assert.equal(sidecar.state, null)
  assert.equal(lifecycle.hasScopedActivity(), false)
  assert.equal(state.closedByDesktop, true)
  assert.equal(closes, 1)
  assert.deepEqual(cleared, [timer])

  let reconnects = 0
  assert.equal(
    scheduleBridgeReconnectIfCurrent({
      lifecycle,
      remoteKey,
      stopping: false,
      enabled: true,
      capturedGlobalGeneration: 4,
      currentGlobalGeneration: 4,
      capturedScopedGeneration: generation,
      scheduleReconnect: () => (reconnects += 1)
    }),
    false
  )
  assert.equal(reconnects, 0)
})

test('disabling one of two profiles sharing a remote keeps the final owner connected', () => {
  const cleared: unknown[] = []
  const lifecycle = new ScopedComputerUseBridgeLifecycle<any>(timer => cleared.push(timer))
  const remoteKey = 'https://shared.example|oauth|profile|current'
  let closes = 0
  const state = { closedByDesktop: false, ws: { close: () => (closes += 1) } }
  const timer = { reconnect: remoteKey } as unknown as ReturnType<typeof setTimeout>
  const pending = Promise.resolve(state)
  const sidecar = { process: {}, startPromise: Promise.resolve(), state: {} }
  let sidecarStops = 0

  const stopSidecar = () => {
    sidecarStops += 1
    sidecar.process = null
    sidecar.startPromise = null
    sidecar.state = null
  }

  lifecycle.acquire(remoteKey, 'pool:research')
  lifecycle.acquire(remoteKey, 'pool:writing')
  lifecycle.connections.set(remoteKey, state)
  lifecycle.connectionPromises.set(remoteKey, pending)
  lifecycle.reconnectTimers.set(remoteKey, timer)
  const generation = lifecycle.generation(remoteKey)

  const release = (owner: string) =>
    releaseBridgeOwnerAndStopSidecarIfIdle({ lifecycle, owner, remoteKey, stopSidecar })

  assert.equal(release('pool:research'), false)
  assert.equal(lifecycle.hasOwners(remoteKey), true)
  assert.equal(lifecycle.connections.get(remoteKey), state)
  assert.equal(lifecycle.connectionPromises.get(remoteKey), pending)
  assert.equal(lifecycle.reconnectTimers.get(remoteKey), timer)
  assert.equal(lifecycle.isCurrent(remoteKey, generation), true)
  assert.equal(closes, 0)
  assert.deepEqual(cleared, [])
  assert.equal(sidecarStops, 0)
  assert.notEqual(sidecar.process, null)

  let reconnects = 0
  assert.equal(
    scheduleBridgeReconnectIfCurrent({
      lifecycle,
      remoteKey,
      stopping: false,
      enabled: true,
      capturedGlobalGeneration: 4,
      currentGlobalGeneration: 4,
      capturedScopedGeneration: generation,
      scheduleReconnect: () => (reconnects += 1)
    }),
    true
  )
  assert.equal(reconnects, 1)

  assert.equal(release('pool:writing'), true)
  assert.equal(lifecycle.hasOwners(remoteKey), false)
  assert.equal(lifecycle.connections.has(remoteKey), false)
  assert.equal(lifecycle.connectionPromises.has(remoteKey), false)
  assert.equal(lifecycle.reconnectTimers.has(remoteKey), false)
  assert.equal(lifecycle.isCurrent(remoteKey, generation), false)
  assert.equal(state.closedByDesktop, true)
  assert.equal(closes, 1)
  assert.deepEqual(cleared, [timer])
  assert.equal(sidecarStops, 1)
  assert.equal(sidecar.process, null)
  assert.equal(sidecar.startPromise, null)
  assert.equal(sidecar.state, null)

  assert.equal(
    scheduleBridgeReconnectIfCurrent({
      lifecycle,
      remoteKey,
      stopping: false,
      enabled: true,
      capturedGlobalGeneration: 4,
      currentGlobalGeneration: 4,
      capturedScopedGeneration: generation,
      scheduleReconnect: () => (reconnects += 1)
    }),
    false
  )
  assert.equal(reconnects, 1)
})

test('connect timeout allows retry while deliberate owner removal suppresses reconnect', () => {
  const lifecycle = new ScopedComputerUseBridgeLifecycle<any>()
  const remoteKey = 'https://remote.example|token|profile|current'
  const owner = 'pool:research'
  let timeoutCloses = 0
  const timedOut = { closedByDesktop: false, ws: { close: () => (timeoutCloses += 1) } }

  lifecycle.acquire(remoteKey, owner)
  lifecycle.connections.set(remoteKey, timedOut)
  lifecycle.closeSockets(remoteKey, true)

  assert.equal(timeoutCloses, 1)
  assert.equal(timedOut.closedByDesktop, false)
  assert.equal(lifecycle.hasOwners(remoteKey), true)

  let removalCloses = 0
  const removed = { closedByDesktop: false, ws: { close: () => (removalCloses += 1) } }
  lifecycle.connections.set(remoteKey, removed)
  lifecycle.release(remoteKey, owner)

  assert.equal(removalCloses, 1)
  assert.equal(removed.closedByDesktop, true)
  assert.equal(lifecycle.hasOwners(remoteKey), false)
})
