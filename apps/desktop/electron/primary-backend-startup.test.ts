import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createBackendConnectionState } from './backend-connection-state'
import { createFirstRunSetupGate } from './first-run-setup-gate'
import { ensureLocalGateway } from './local-gateway'
import {
  createPrimaryRemoteConnection,
  FirstRunSetupResetError,
  runPrimaryBackendStartup
} from './primary-backend-startup'

const bootstrapBackend = {
  activeRoot: '/tmp/hermes-home/hermes-agent',
  kind: 'bootstrap-needed',
  platform: 'linux'
}

function startupOptions(overrides: Record<string, unknown> = {}) {
  return {
    assertCurrentAttempt: () => {},
    connectRemote: vi.fn(async remote => ({ baseUrl: remote.baseUrl, mode: 'remote' as const })),
    ensureLocalRuntime: vi.fn(async backend => ({ ...backend, command: 'hermes' })),
    prepareLocalBackend: vi.fn(async () => bootstrapBackend),
    resolveRemote: vi.fn(async () => null),
    waitForDecision: vi.fn(async () => 'continue-local' as const),
    waitForLocalStart: vi.fn(async () => {}),
    ...overrides
  }
}

test('primary remote descriptor preserves a resolved registry connection id', () => {
  const connection = createPrimaryRemoteConnection(
    {
      authMode: 'token',
      baseUrl: 'https://gateway.example.com',
      connectionId: 'skateway',
      remoteKind: 'url',
      source: 'settings',
      token: 'secret',
      wsUrl: 'wss://gateway.example.com/api/ws'
    },
    ['ready'],
    { isFullscreen: false }
  )

  assert.equal(connection.connectionId, 'skateway')
  assert.equal(connection.mode, 'remote')
  assert.deepEqual(connection.logs, ['ready'])
  assert.equal(connection.isFullscreen, false)
})

test('primary remote descriptor preserves the effective SSH dialing identity', () => {
  const ssh = {
    effectiveConfigFingerprint: 'effective-config',
    host: 'build-host',
    remoteHermesPath: '/srv/hermes',
    remoteProfile: 'default',
    user: 'alice'
  }

  const connection = createPrimaryRemoteConnection(
    {
      baseUrl: 'http://127.0.0.1:49152',
      remoteKind: 'ssh',
      ssh,
      token: 'secret',
      wsUrl: 'ws://127.0.0.1:49152/api/ws'
    },
    [],
    {}
  )

  assert.equal(connection.ssh, ssh)
  assert.equal(connection.ssh?.effectiveConfigFingerprint, 'effective-config')
})

test('primary remote descriptor keeps legacy unregistered routes unqualified', () => {
  const connection = createPrimaryRemoteConnection(
    {
      baseUrl: 'https://env.example.com',
      source: 'env',
      token: 'secret',
      wsUrl: 'wss://env.example.com/api/ws'
    },
    [],
    {}
  )

  assert.equal('connectionId' in connection, false)
})

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

const localStartupPhases = [
  ['resolveRemote', null],
  ['waitForLocalStart', undefined],
  ['prepareLocalBackend', bootstrapBackend],
  ['waitForDecision', 'continue-local'],
  ['ensureLocalRuntime', { ...bootstrapBackend, command: 'hermes' }]
] as const

test.each(localStartupPhases)('invalidating pending %s prevents gateway ensure and descriptor publication', async (phase, value) => {
  const state = createBackendConnectionState()
  const attempt = state.startAttempt()
  let entered!: () => void
  let release!: () => void
  const paused = new Promise<void>(resolve => { entered = resolve })
  const resumed = new Promise<void>(resolve => { release = resolve })
  const endpoint = { profile_id: '/private/profile', instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none' }
  const runEnsure = vi.fn(async () => ({ code: 0, stdout: JSON.stringify({ state: 'ready', endpoint }) }))
  const publish = vi.fn(connection => connection)

  const options = startupOptions({
    assertCurrentAttempt: () => state.assertCurrentAttempt(attempt),
    signal: new AbortController().signal,
    [phase]: vi.fn(async () => {
      entered()
      await resumed

      return value
    })
  })

  // Cross the same setup-to-canonical-ensure seam as the primary caller.
  const pending = runPrimaryBackendStartup(options).then(setup => {
    assert.equal(setup.kind, 'local')

    return ensureLocalGateway(runEnsure)
  }).then(publish)

  state.setPromise(attempt, pending)
  await paused
  state.invalidate()
  release()
  await pending.catch(() => {})
  assert.deepEqual({ ensures: runEnsure.mock.calls.length, publications: publish.mock.calls.length }, { ensures: 0, publications: 0 })
  await assert.rejects(pending, /superseded by a newer connection attempt/)

  for (const [nextPhase] of localStartupPhases.slice(localStartupPhases.findIndex(([name]) => name === phase) + 1)) {
    assert.equal(options[nextPhase].mock.calls.length, 0, `stale ${phase} must not enter ${nextPhase}`)
  }
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
