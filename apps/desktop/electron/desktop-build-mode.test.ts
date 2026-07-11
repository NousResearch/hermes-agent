import assert from 'node:assert/strict'
import test from 'node:test'

import {
  assertConnectionModeAllowed,
  createRemoteConnectionGate,
  shouldResumeRemoteConnectionGate
} from './desktop-build-mode'

test('remote-only builds reject local connection mode', () => {
  assert.throws(
    () => assertConnectionModeAllowed('local', true),
    /requires a remote Hermes connection/
  )
  assert.doesNotThrow(() => assertConnectionModeAllowed('remote', true))
  assert.doesNotThrow(() => assertConnectionModeAllowed('cloud', true))
  assert.doesNotThrow(() => assertConnectionModeAllowed('local', false))
})

test('remote connection gate shares one waiter and resumes it once', async () => {
  const gate = createRemoteConnectionGate()
  const first = gate.wait()
  const second = gate.wait()

  assert.equal(first, second)
  assert.equal(gate.hasWaiter(), true)

  gate.resume()
  await first

  assert.equal(gate.hasWaiter(), false)
})

test('only a global remote apply resumes first-run setup', () => {
  assert.equal(shouldResumeRemoteConnectionGate(true, null, true), true)
  assert.equal(shouldResumeRemoteConnectionGate(true, 'work', true), false)
  assert.equal(shouldResumeRemoteConnectionGate(true, null, false), false)
  assert.equal(shouldResumeRemoteConnectionGate(false, null, true), false)
})
