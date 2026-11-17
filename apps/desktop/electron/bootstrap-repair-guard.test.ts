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
})

test('soft restart budget exhausts at maxSoftAttempts+1 and escalates', () => {
  const decision = decideBootstrapRepair({
    attempt: 4,
    maxSoftAttempts: 3,
    primaryBackendAlive: true
  })

  assert.equal(decision.hardReinstall, true)
  assert.equal(decision.attempt, 4)
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
