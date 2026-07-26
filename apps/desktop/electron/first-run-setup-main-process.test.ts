import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { applyConnectionChange } from './connection-apply'
import { createFirstRunSetupGate } from './first-run-setup-gate'
import { runPrimaryBackendStartup } from './primary-backend-startup'
import { rehomePrimaryConnection } from './primary-connection-rehome'

test('a first-run bootstrap-needed remote apply connects without ensuring or bootstrapping locally', async () => {
  const gate = createFirstRunSetupGate({ stuckAfterMs: 0 })

  const bootstrapBackend = {
    activeRoot: '/tmp/hermes-home/hermes-agent',
    kind: 'bootstrap-needed',
    platform: 'linux'
  }

  const candidateRemote = {
    authMode: 'token',
    baseUrl: 'https://gateway.example.com/hermes',
    source: 'settings',
    token: 'secret',
    wsUrl: 'wss://gateway.example.com/hermes/api/ws?token=secret'
  }

  let savedRemote: typeof candidateRemote | null = null

  const resolveRemote = vi.fn(async () => savedRemote)
  const connectRemote = vi.fn(async remote => ({ ...remote, mode: 'remote' as const }))
  const runBootstrap = vi.fn()

  const ensureLocalRuntime = vi.fn(async backend => {
    await runBootstrap()

    return { ...backend, command: 'hermes' }
  })

  const teardownPrimaryBackend = vi.fn(async () => {})
  const cancelSshBootstrap = vi.fn(async () => {})
  const teardownSsh = vi.fn(async () => {})
  const clearLocalBootstrapFailure = vi.fn()
  const notifyConnectionApplied = vi.fn()
  const waitForLocalStart = vi.fn(async () => {})
  const prepareLocalBackend = vi.fn(async () => bootstrapBackend)

  const pendingConnection = runPrimaryBackendStartup({
    connectRemote,
    ensureLocalRuntime,
    prepareLocalBackend,
    resolveRemote,
    waitForDecision: gate.wait,
    waitForLocalStart
  })

  await vi.waitFor(() => assert.equal(gate.hasWaiter(), true))

  // Mirrors the IPC handler's production ordering: persist the tested config,
  // then re-home. The pending start must re-resolve this saved value.
  savedRemote = candidateRemote

  await applyConnectionChange({
    cancelAndWait: cancelSshBootstrap,
    isPrimary: true,
    rehomePrimary: () =>
      rehomePrimaryConnection({
        clearLocalBootstrapFailure,
        mode: 'remote',
        notifyConnectionApplied,
        resumeFirstRunRemote: gate.abandonForRemoteApply,
        teardownPrimaryBackend
      }),
    scope: '',
    sendApplied: notifyConnectionApplied,
    stopPool: vi.fn(),
    teardownPrimary: teardownPrimaryBackend,
    teardownSsh
  })

  assert.deepEqual(await pendingConnection, {
    kind: 'remote',
    connection: { ...candidateRemote, mode: 'remote' }
  })
  assert.deepEqual(resolveRemote.mock.calls, [[], []])
  assert.deepEqual(connectRemote.mock.calls, [[candidateRemote]])
  assert.deepEqual(waitForLocalStart.mock.calls, [[]])
  assert.deepEqual(prepareLocalBackend.mock.calls, [[]])
  assert.equal(ensureLocalRuntime.mock.calls.length, 0)
  assert.equal(runBootstrap.mock.calls.length, 0)
  assert.deepEqual(cancelSshBootstrap.mock.calls, [['']])
  assert.deepEqual(teardownSsh.mock.calls, [['']])
  assert.equal(teardownPrimaryBackend.mock.calls.length, 0)
  assert.equal(clearLocalBootstrapFailure.mock.calls.length, 1)
  assert.equal(notifyConnectionApplied.mock.calls.length, 0)
})

test('a primary apply without an active first-run gate tears down before reconnect notification', async () => {
  const order: string[] = []
  const clearLocalBootstrapFailure = vi.fn(() => order.push('clear-failure'))

  const teardownPrimaryBackend = vi.fn(async () => {
    order.push('teardown')
  })

  const notifyConnectionApplied = vi.fn(() => order.push('notify'))

  assert.deepEqual(
    await rehomePrimaryConnection({
      clearLocalBootstrapFailure,
      mode: 'remote',
      notifyConnectionApplied,
      resumeFirstRunRemote: () => false,
      teardownPrimaryBackend
    }),
    { resumedFirstRunRemote: false }
  )
  assert.deepEqual(teardownPrimaryBackend.mock.calls, [[{ soft: true }]])
  assert.deepEqual(order, ['clear-failure', 'teardown', 'notify'])
})
