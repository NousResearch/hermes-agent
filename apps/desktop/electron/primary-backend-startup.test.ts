import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createFirstRunSetupGate } from './first-run-setup-gate'
import { FirstRunSetupResetError, runPrimaryBackendStartup } from './primary-backend-startup'

const bootstrapBackend = {
  activeRoot: '/tmp/hermes-home/hermes-agent',
  kind: 'bootstrap-needed',
  platform: 'linux'
}

function startupOptions(overrides: Record<string, unknown> = {}) {
  return {
    connectRemote: vi.fn(async remote => ({ baseUrl: remote.baseUrl, mode: 'remote' as const })),
    ensureLocalRuntime: vi.fn(async backend => ({ ...backend, command: 'hermes' })),
    prepareLocalBackend: vi.fn(async () => bootstrapBackend),
    resolveRemote: vi.fn(async () => null),
    waitForDecision: vi.fn(async () => 'continue-local' as const),
    waitForLocalStart: vi.fn(async () => {}),
    ...overrides
  }
}

test('remote apply re-resolves the saved connection without ensuring a local runtime', async () => {
  const gate = createFirstRunSetupGate({ stuckAfterMs: 0 })
  const savedRemote = { baseUrl: 'https://gateway.example.com/hermes' }
  let configuredRemote: typeof savedRemote | null = null

  const options = startupOptions({
    resolveRemote: vi.fn(async () => configuredRemote),
    waitForDecision: gate.wait
  })

  const pending = runPrimaryBackendStartup(options)

  await vi.waitFor(() => assert.equal(gate.hasWaiter(), true))
  configuredRemote = savedRemote
  assert.equal(gate.abandonForRemoteApply(), true)

  assert.deepEqual(await pending, {
    kind: 'remote',
    connection: { baseUrl: savedRemote.baseUrl, mode: 'remote' }
  })
  assert.deepEqual(options.resolveRemote.mock.calls, [[], []])
  assert.deepEqual(options.connectRemote.mock.calls, [[savedRemote]])
  assert.equal(options.ensureLocalRuntime.mock.calls.length, 0)
})

test('an already-saved remote bypasses every local startup step', async () => {
  const savedRemote = { baseUrl: 'https://gateway.example.com/hermes' }
  const options = startupOptions({ resolveRemote: vi.fn(async () => savedRemote) })

  assert.deepEqual(await runPrimaryBackendStartup(options), {
    kind: 'remote',
    connection: { baseUrl: savedRemote.baseUrl, mode: 'remote' }
  })
  assert.equal(options.waitForLocalStart.mock.calls.length, 0)
  assert.equal(options.prepareLocalBackend.mock.calls.length, 0)
  assert.equal(options.waitForDecision.mock.calls.length, 0)
  assert.equal(options.ensureLocalRuntime.mock.calls.length, 0)
})

test('remote apply fails clearly when no saved remote can be resolved', async () => {
  const gate = createFirstRunSetupGate({ stuckAfterMs: 0 })
  const options = startupOptions({ waitForDecision: gate.wait })
  const pending = runPrimaryBackendStartup(options)

  await vi.waitFor(() => assert.equal(gate.hasWaiter(), true))
  gate.abandonForRemoteApply()

  await assert.rejects(pending, /without a saved remote backend/)
  assert.equal(options.connectRemote.mock.calls.length, 0)
  assert.equal(options.ensureLocalRuntime.mock.calls.length, 0)
})

test('continue local waits for update exclusion and ensures the prepared runtime exactly once', async () => {
  const gate = createFirstRunSetupGate({ stuckAfterMs: 0 })
  const runtimeBackend = { ...bootstrapBackend, command: 'hermes' }

  const options = startupOptions({
    ensureLocalRuntime: vi.fn(async () => runtimeBackend),
    waitForDecision: gate.wait
  })

  const pending = runPrimaryBackendStartup(options)

  await vi.waitFor(() => assert.equal(gate.hasWaiter(), true))
  gate.continueLocal()

  assert.deepEqual(await pending, { kind: 'local', backend: runtimeBackend })
  assert.deepEqual(options.waitForLocalStart.mock.calls, [[]])
  assert.deepEqual(options.prepareLocalBackend.mock.calls, [[]])
  assert.deepEqual(options.ensureLocalRuntime.mock.calls, [[bootstrapBackend]])
  assert.deepEqual(options.resolveRemote.mock.calls, [[]])
})

test('reset rejects with a typed error and never enters either backend', async () => {
  const gate = createFirstRunSetupGate({ stuckAfterMs: 0 })
  const options = startupOptions({ waitForDecision: gate.wait })
  const pending = runPrimaryBackendStartup(options)

  await vi.waitFor(() => assert.equal(gate.hasWaiter(), true))
  gate.resetForRetry()

  await assert.rejects(pending, error => error instanceof FirstRunSetupResetError && error.firstRunSetupReset)
  assert.equal(options.connectRemote.mock.calls.length, 0)
  assert.equal(options.ensureLocalRuntime.mock.calls.length, 0)
})
