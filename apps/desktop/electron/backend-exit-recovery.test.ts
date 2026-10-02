import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createBackendConnectionState } from './backend-connection-state'
import { createBackendExitRecoveryLatch } from './backend-exit-recovery'

type Child = { pid: number }

// Mirrors main.ts::runHermesStart's exit handler: the slot state the handler
// reads when the child's exit is classified as stale (#112344).
function slotState(state: ReturnType<typeof createBackendConnectionState<Child, unknown>>, extra = {}) {
  return {
    hasCurrentOwner: state.getProcess() !== null || state.getPromise() !== null,
    hasPendingStart: false,
    intentionalTeardown: false,
    ...extra
  }
}

test('a stale exit that leaves the primary slot empty is claimed once, then re-armed by the next ready backend', () => {
  const state = createBackendConnectionState<Child, unknown>()
  const latch = createBackendExitRecoveryLatch()
  const attempt = state.startAttempt()
  state.setPromise(attempt, Promise.resolve({ mode: 'local' }))
  const owner = state.attachProcess(attempt, { pid: 100 })!

  // The slot is emptied (invalidate without a follow-up start, or the child's
  // own `error` handler cleared it first) and only THEN the exit lands.
  state.invalidate()
  assert.equal(state.clearForCurrentProcess(owner), false, 'exit is classified stale')

  // Base behaviour was "log and return" here; the supervisor now owns the respawn.
  assert.equal(latch.claim(slotState(state)), true)
  // A second stale event for the same empty slot (error + exit pair) coalesces.
  assert.equal(latch.claim(slotState(state)), false)

  // The replacement becomes ready: the latch re-arms for the next death.
  latch.reset()
  assert.equal(latch.claim(slotState(state)), true)
})

test('a stale exit is not claimed while the slot has an owner, a start is pending, or teardown is intentional', () => {
  const state = createBackendConnectionState<Child, unknown>()
  const latch = createBackendExitRecoveryLatch()
  const first = state.startAttempt()
  state.setPromise(first, Promise.resolve({ mode: 'local' }))
  const oldOwner = state.attachProcess(first, { pid: 100 })!

  // Re-home: invalidate, then a replacement attempt publishes before the old exit lands.
  state.invalidate()
  const replacement = state.startAttempt()
  state.setPromise(replacement, new Promise(() => {}))
  assert.equal(state.clearForCurrentProcess(oldOwner), false)
  assert.equal(latch.claim(slotState(state)), false, 'published replacement attempt owns the slot')

  // A remote descriptor with no child process is still an owner.
  const remote = createBackendConnectionState<Child, unknown>()
  const remoteAttempt = remote.startAttempt()
  remote.setPromise(remoteAttempt, Promise.resolve({ mode: 'remote' }))
  assert.equal(latch.claim(slotState(remote)), false)

  const empty = createBackendConnectionState<Child, unknown>()
  assert.equal(latch.claim(slotState(empty, { hasPendingStart: true })), false)
  assert.equal(latch.claim(slotState(empty, { intentionalTeardown: true })), false)
  // Nothing above consumed the latch.
  assert.equal(latch.claim(slotState(empty)), true)
})

test('a backend that dies after every ready is respawned at most maxRespawns times per window, then reported as crash-looping', () => {
  let clock = 1_000
  const latch = createBackendExitRecoveryLatch({ maxRespawns: 3, windowMs: 120_000, now: () => clock })
  const empty = { hasCurrentOwner: false, hasPendingStart: false, intentionalTeardown: false }

  // ready -> dies -> respawn, three times within the window.
  for (let i = 0; i < 3; i++) {
    assert.equal(latch.claim(empty), true, `respawn ${i + 1}`)
    assert.equal(latch.isCrashLooping(), false)
    latch.reset()
    clock += 5_000
  }

  // The fourth death inside the window is a crash loop: no respawn.
  assert.equal(latch.claim(empty), false)
  assert.equal(latch.isCrashLooping(), true)

  // Once the window has passed the supervisor may try again.
  clock += 120_000
  assert.equal(latch.claim(empty), true)
  assert.equal(latch.isCrashLooping(), false)
})

test('a claimed recovery that fails before ready can retry within the same crash-loop budget', () => {
  let clock = 1_000
  const latch = createBackendExitRecoveryLatch({ maxRespawns: 3, windowMs: 120_000, now: () => clock })
  const empty = { hasCurrentOwner: false, hasPendingStart: false, intentionalTeardown: false }

  assert.equal(latch.claim(empty), true, 'ready backend death grants the first recovery')
  clock += 1_000
  assert.equal(latch.retryAfterFailedStart(empty), true, 'pre-ready failure grants a bounded retry')
  clock += 1_000
  assert.equal(latch.retryAfterFailedStart(empty), true, 'the final budgeted retry is still admitted')
  clock += 1_000
  assert.equal(latch.retryAfterFailedStart(empty), false, 'a fourth recovery attempt is refused')
  assert.equal(latch.isCrashLooping(), true)
})

test('a failed recovery does not release its claim while another owner/start or teardown is present', () => {
  const latch = createBackendExitRecoveryLatch()
  const empty = { hasCurrentOwner: false, hasPendingStart: false, intentionalTeardown: false }

  assert.equal(latch.claim(empty), true)
  assert.equal(latch.retryAfterFailedStart({ ...empty, hasPendingStart: true }), false)
  assert.equal(latch.claim(empty), false, 'the pending-start refusal preserves the original claim')

  latch.reset()
  assert.equal(latch.claim(empty), true)
  assert.equal(latch.retryAfterFailedStart({ ...empty, hasCurrentOwner: true }), false)
  assert.equal(latch.claim(empty), false, 'the current-owner refusal preserves the original claim')

  latch.reset()
  assert.equal(latch.claim(empty), true)
  assert.equal(latch.retryAfterFailedStart({ ...empty, intentionalTeardown: true }), false)
  assert.equal(latch.claim(empty), false, 'intentional teardown does not re-arm recovery')
})

test('an attached backend torn down as unexpected (dead or drifted token) is claimed and a respawn is scheduled (#121988)', () => {
  const state = createBackendConnectionState<Child, unknown>()
  const latch = createBackendExitRecoveryLatch()
  const attempt = state.startAttempt()
  // An attached backend spawns no child (main.ts: "Nothing was spawned, so
  // there is no child to own"); the slot holds only the resolved connection.
  state.setPromise(attempt, Promise.resolve({ mode: 'local', attached: true }))

  // Mirrors startAttachedBackendMonitor's catch handler: invalidate the slot,
  // then immediately try to claim the respawn in the same tick.
  state.invalidate()
  assert.equal(
    latch.claim(slotState(state)),
    true,
    'an unexpected attach teardown must not read as intentional, or the respawn is silently dropped'
  )

  // The bug this guards: invalidating through a path that also marks the
  // teardown intentional (as `invalidatePrimaryConnection()` does, for the
  // deliberate re-home/quit/config-apply cases) makes the very next claim()
  // above refuse the respawn, leaving the app with no backend.
  const otherState = createBackendConnectionState<Child, unknown>()
  const otherLatch = createBackendExitRecoveryLatch()
  const otherAttempt = otherState.startAttempt()
  otherState.setPromise(otherAttempt, Promise.resolve({ mode: 'local', attached: true }))
  otherState.invalidate()
  assert.equal(
    otherLatch.claim(slotState(otherState, { intentionalTeardown: true })),
    false,
    'marking an unexpected attach teardown intentional would silently drop the respawn'
  )
})

test('a failed start that owns no recovery claim is not re-armed and spends no budget', () => {
  let clock = 1_000
  const latch = createBackendExitRecoveryLatch({ maxRespawns: 3, windowMs: 120_000, now: () => clock })
  const empty = { hasCurrentOwner: false, hasPendingStart: false, intentionalTeardown: false }

  // Nothing claimed yet (fresh latch): a pre-ready failure of a user-driven
  // start is not the supervisor's retry to take.
  assert.equal(latch.retryAfterFailedStart(empty), false, 'fresh latch has no claim to re-arm')
  assert.equal(latch.isCrashLooping(), false)

  // A ready backend released the claim: a later failed start still owns nothing.
  assert.equal(latch.claim(empty), true)
  latch.reset()
  clock += 1_000
  assert.equal(latch.retryAfterFailedStart(empty), false, 'reset() leaves nothing to re-arm')
  assert.equal(latch.isCrashLooping(), false)

  // Neither refusal consumed the window: the remaining two grants are intact.
  clock += 1_000
  assert.equal(latch.claim(empty), true, 'second respawn')
  latch.reset()
  clock += 1_000
  assert.equal(latch.claim(empty), true, 'third respawn')
  latch.reset()
  clock += 1_000
  assert.equal(latch.claim(empty), false, 'fourth is the real budget exhaustion')
  assert.equal(latch.isCrashLooping(), true)
})
