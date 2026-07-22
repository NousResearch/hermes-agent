/**
 * Tests for electron/first-run-gate.ts.
 *
 * Run with: vitest run --project electron electron/first-run-gate.test.ts
 * (Auto-discovered by the electron vitest project; runs in npm test:desktop:platforms.)
 *
 * The pure state machine behind the first-run "install vs connect" choice:
 * entering/leaving the waiting state, idempotent waitForDecision, choose/abort
 * resolution, onChanged firing exactly once per transition, reset-after-decision,
 * and no-op choose/abort when not waiting.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createFirstRunGate } from './first-run-gate'

test('a fresh gate is not required and reports required:false', () => {
  const gate = createFirstRunGate()
  assert.deepEqual(gate.state(), { required: false })
})

test('waitForDecision enters the waiting state and fires onChanged once', () => {
  const events: Array<{ required: boolean }> = []
  const gate = createFirstRunGate(state => events.push(state))

  gate.waitForDecision()

  assert.deepEqual(gate.state(), { required: true })
  assert.deepEqual(events, [{ required: true }])
})

test('waitForDecision is idempotent: same promise, no extra onChanged', () => {
  const events: Array<{ required: boolean }> = []
  const gate = createFirstRunGate(state => events.push(state))

  const first = gate.waitForDecision()
  const second = gate.waitForDecision()

  assert.equal(first, second)
  assert.deepEqual(events, [{ required: true }])
})

test('chooseInstall resolves the wait with install and clears required', async () => {
  const events: Array<{ required: boolean }> = []
  const gate = createFirstRunGate(state => events.push(state))

  const decision = gate.waitForDecision()
  gate.chooseInstall()

  assert.equal(await decision, 'install')
  assert.deepEqual(gate.state(), { required: false })
  assert.deepEqual(events, [{ required: true }, { required: false }])
})

test('abort resolves the wait with abort and clears required', async () => {
  const events: Array<{ required: boolean }> = []
  const gate = createFirstRunGate(state => events.push(state))

  const decision = gate.waitForDecision()
  gate.abort()

  assert.equal(await decision, 'abort')
  assert.deepEqual(gate.state(), { required: false })
  assert.deepEqual(events, [{ required: true }, { required: false }])
})

test('the gate resets after a decision so a later wait asks again', async () => {
  const events: Array<{ required: boolean }> = []
  const gate = createFirstRunGate(state => events.push(state))

  const first = gate.waitForDecision()
  gate.chooseInstall()
  assert.equal(await first, 'install')

  // A second bootstrap-needed (e.g. retry after reload) must re-enter the wait.
  const second = gate.waitForDecision()
  assert.notEqual(first, second)
  assert.deepEqual(gate.state(), { required: true })

  gate.abort()
  assert.equal(await second, 'abort')

  assert.deepEqual(events, [{ required: true }, { required: false }, { required: true }, { required: false }])
})

test('chooseInstall is a no-op when not waiting (no throw, no onChanged)', () => {
  const events: Array<{ required: boolean }> = []
  const gate = createFirstRunGate(state => events.push(state))

  assert.doesNotThrow(() => gate.chooseInstall())
  assert.deepEqual(gate.state(), { required: false })
  assert.deepEqual(events, [])
})

test('abort is a no-op when not waiting (no throw, no onChanged)', () => {
  const events: Array<{ required: boolean }> = []
  const gate = createFirstRunGate(state => events.push(state))

  assert.doesNotThrow(() => gate.abort())
  assert.deepEqual(gate.state(), { required: false })
  assert.deepEqual(events, [])
})

test('a second settle after a decision is a no-op (idle latch)', async () => {
  const events: Array<{ required: boolean }> = []
  const gate = createFirstRunGate(state => events.push(state))

  const decision = gate.waitForDecision()
  gate.chooseInstall()
  gate.abort() // late/duplicate: nothing is waiting anymore

  assert.equal(await decision, 'install')
  assert.deepEqual(events, [{ required: true }, { required: false }])
})

test('the gate works without an onChanged callback', async () => {
  const gate = createFirstRunGate()

  const decision = gate.waitForDecision()
  assert.deepEqual(gate.state(), { required: true })
  gate.chooseInstall()

  assert.equal(await decision, 'install')
})
