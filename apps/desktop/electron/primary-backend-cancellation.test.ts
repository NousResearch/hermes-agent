import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createFirstRunSetupGate } from './first-run-setup-gate'
import { runPrimaryBackendStartup } from './primary-backend-startup'

test('quit cancels first-run startup without entering the installer', async () => {
  const controller = new AbortController()
  const gate = createFirstRunSetupGate({ stuckAfterMs: 0 })
  let installed = false

  const startup = runPrimaryBackendStartup({
    signal: controller.signal,
    resolveRemote: async () => null,
    connectRemote: async () => ({}),
    waitForLocalStart: async () => {},
    prepareLocalBackend: () => ({ kind: 'bootstrap-needed' }),
    waitForDecision: backend => gate.wait(backend),
    ensureLocalRuntime: async () => {
      installed = true

      return {}
    }
  })

  for (let i = 0; i < 20; i++) {
    await Promise.resolve()
  }

  assert.equal(gate.hasWaiter(), true)
  let settled = false
  void startup.catch(() => {
    settled = true
  })
  controller.abort()

  for (let i = 0; i < 20; i++) {
    await Promise.resolve()
  }

  assert.equal(settled, true, 'quit must not wait for a user choice')
  gate.continueLocal()
  await startup.catch(() => {})
  assert.equal(installed, false, 'a late choice must not launch the installer')
})
