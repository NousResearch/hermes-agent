import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createFirstRunSetupGate } from './first-run-setup-gate'

const bootstrapBackend = {
  activeRoot: '/tmp/hermes-home/hermes-agent',
  kind: 'bootstrap-needed',
  platform: 'linux'
}

function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function settledState(promise: Promise<unknown>) {
  return Promise.race([promise.then(() => 'resolved'), delay(10).then(() => 'pending')])
}

test('first-run setup gate skips non-bootstrap backends', async () => {
  const prompts = []
  const gate = createFirstRunSetupGate({ promptChoice: backend => prompts.push(backend), stuckAfterMs: 0 })

  await gate.wait({ kind: 'remote' })
  await gate.wait(null)

  assert.deepEqual(prompts, [])
  assert.equal(gate.hasWaiter(), false)
})

test('first-run setup gate prompts once for concurrent waits', async () => {
  const prompts = []
  const gate = createFirstRunSetupGate({ promptChoice: backend => prompts.push(backend), stuckAfterMs: 0 })

  const first = gate.wait(bootstrapBackend)
  const second = gate.wait(bootstrapBackend)

  assert.equal(gate.hasWaiter(), true)
  assert.equal(prompts.length, 1)
  assert.equal(await settledState(first), 'pending')

  gate.continueLocal()

  assert.deepEqual(await Promise.all([first, second]), ['continue-local', 'continue-local'])
  assert.equal(gate.hasWaiter(), false)
  assert.equal(gate.isLocalBootstrapConfirmed(), true)
})

test('continueLocal keeps the setup choice visible until bootstrap owns the overlay', async () => {
  let hidden = 0
  const gate = createFirstRunSetupGate({ hideChoice: () => hidden++, stuckAfterMs: 0 })
  const pending = gate.wait(bootstrapBackend)

  gate.continueLocal()

  assert.equal(await pending, 'continue-local')
  assert.equal(hidden, 0)
  assert.equal(gate.isLocalBootstrapConfirmed(), true)
})

test('retry reset preserves the local install confirmation', async () => {
  const prompts = []
  const gate = createFirstRunSetupGate({ promptChoice: backend => prompts.push(backend), stuckAfterMs: 0 })

  const pending = gate.wait(bootstrapBackend)
  gate.continueLocal()
  await pending

  gate.resetForRetry()
  await gate.wait(bootstrapBackend)

  assert.equal(gate.isLocalBootstrapConfirmed(), true)
  assert.equal(prompts.length, 1)
  assert.equal(gate.hasWaiter(), false)
})

test('retry reset explicitly settles an active waiter without allowing local bootstrap', async () => {
  const gate = createFirstRunSetupGate({ stuckAfterMs: 0 })
  const pending = gate.wait(bootstrapBackend)

  gate.resetForRetry()

  assert.equal(await pending, 'reset')
  assert.equal(gate.hasWaiter(), false)
  assert.equal(gate.isLocalBootstrapConfirmed(), false)
})

test('repair reset clears the local install confirmation and shows the gate again', async () => {
  const prompts = []
  const gate = createFirstRunSetupGate({ promptChoice: backend => prompts.push(backend), stuckAfterMs: 0 })

  const pending = gate.wait(bootstrapBackend)
  gate.continueLocal()
  await pending

  gate.resetForRepair()
  const next = gate.wait(bootstrapBackend)

  assert.equal(gate.isLocalBootstrapConfirmed(), false)
  assert.equal(prompts.length, 2)
  assert.equal(gate.hasWaiter(), true)

  gate.continueLocal()
  await next
})

test('remote apply settles the gated boot for remote re-resolution and hides the choice', async () => {
  let hidden = 0
  const gate = createFirstRunSetupGate({ hideChoice: () => hidden++, stuckAfterMs: 0 })
  const pending = gate.wait(bootstrapBackend)

  const resumedWaiter = gate.abandonForRemoteApply()

  assert.equal(resumedWaiter, true)
  assert.equal(hidden, 1)
  assert.equal(gate.hasWaiter(), false)
  assert.equal(gate.isLocalBootstrapConfirmed(), false)
  assert.equal(await pending, 'remote-applied')
})

test('remote apply without a waiter has no first-run side effects', async () => {
  let hidden = 0
  const gate = createFirstRunSetupGate({ hideChoice: () => hidden++, stuckAfterMs: 0 })
  const pending = gate.wait(bootstrapBackend)

  gate.continueLocal()
  await pending

  assert.equal(gate.abandonForRemoteApply(), false)
  assert.equal(hidden, 0)
  assert.equal(gate.isLocalBootstrapConfirmed(), true)
})
