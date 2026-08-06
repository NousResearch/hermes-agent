import assert from 'node:assert/strict'

import { test } from 'vitest'

import { decideBootstrapRepair } from './bootstrap-repair-guard'

test('first soft attempt with alive backend returns soft restart', () => {
  const decision = decideBootstrapRepair({
    attempt: 1,
    primaryBackendAlive: true
  })

  assert.equal(decision.hardReinstall, false)
  assert.equal(decision.attempt, 1)
  assert.match(decision.reason, /still alive/)
  assert.match(decision.reason, /1\/3/)
})

test('first attempt with dead backend still returns soft restart', () => {
  const decision = decideBootstrapRepair({
    attempt: 1,
    primaryBackendAlive: false
  })

  assert.equal(decision.hardReinstall, false)
  assert.match(decision.reason, /has exited/)
})

test('soft restart budget exhausts at maxSoftAttempts+1 and escalates', () => {
  const decision = decideBootstrapRepair({
    attempt: 4,
    maxSoftAttempts: 3,
    primaryBackendAlive: true
  })

  assert.equal(decision.hardReinstall, true)
  assert.equal(decision.attempt, 4)
  assert.match(decision.reason, /exceeds soft-restart budget/)
})

test('attempt exactly at maxSoftAttempts is still soft', () => {
  const decision = decideBootstrapRepair({
    attempt: 3,
    maxSoftAttempts: 3,
    primaryBackendAlive: true
  })

  assert.equal(decision.hardReinstall, false)
  assert.equal(decision.attempt, 3)
})

test('custom maxSoftAttempts is honored', () => {
  const soft = decideBootstrapRepair({
    attempt: 5,
    maxSoftAttempts: 10,
    primaryBackendAlive: true
  })

  assert.equal(soft.hardReinstall, false)

  const hard = decideBootstrapRepair({
    attempt: 11,
    maxSoftAttempts: 10,
    primaryBackendAlive: false
  })

  assert.equal(hard.hardReinstall, true)
})

test('default maxSoftAttempts is 3', () => {
  // Probe the default indirectly: attempt 4 with no override must escalate.
  const decision = decideBootstrapRepair({
    attempt: 4,
    primaryBackendAlive: true
  })

  assert.equal(decision.hardReinstall, true)
})

test('fractional or zero attempts are clamped to 1', () => {
  const zeroDecision = decideBootstrapRepair({
    attempt: 0,
    primaryBackendAlive: true
  })

  assert.equal(zeroDecision.attempt, 1)
  assert.equal(zeroDecision.hardReinstall, false)

  const fractionalDecision = decideBootstrapRepair({
    attempt: 2.7,
    primaryBackendAlive: true
  })

  assert.equal(fractionalDecision.attempt, 2)
  assert.equal(fractionalDecision.hardReinstall, false)
})

test('alive=false on a high attempt number still escalates (defense in depth)', () => {
  // A dead backend should normally be handled by the renderer before it
  // reaches the repair path, but if it does reach us with a high attempt
  // count we still escalate — never silently keep soft-restarting.
  const decision = decideBootstrapRepair({
    attempt: 5,
    maxSoftAttempts: 3,
    primaryBackendAlive: false
  })

  assert.equal(decision.hardReinstall, true)
})
