import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createBackendConnectionState } from './backend-connection-state'
import { createFirstRunSetupGate } from './first-run-setup-gate'
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

function startupOptions<T extends Record<string, unknown> = Record<string, never>>(overrides: T = {} as T) {
  const base = {
    assertCurrentAttempt: () => {},
    connectRemote: vi.fn(async remote => ({ baseUrl: remote.baseUrl, mode: 'remote' as const })),
    ensureLocalRuntime: vi.fn(async backend => ({ ...backend, command: 'hermes' })),
    prepareLocalBackend: vi.fn(async () => bootstrapBackend),
    resolveRemote: vi.fn(async () => null),
    waitForDecision: vi.fn(async () => 'continue-local' as const),
    waitForLocalStart: vi.fn(async () => {})
  }
  return { ...base, ...overrides } as typeof base & T
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

test('primary remote descriptor preserves the gateway extra headers for REST calls', () => {
  // Chat and the Test button carry the headers via the exact-URL WS store, but
  // fetchJsonForBackend reads descriptor.headers — dropping them here made every
  // Settings/session-history call hit an access proxy unauthenticated (#112072).
  const headers = { 'CF-Access-Client-Id': 'client-id', 'CF-Access-Client-Secret': 'client-secret' }

  const connection = createPrimaryRemoteConnection(
    {
      authMode: 'token',
      baseUrl: 'https://gateway.example.com',
      connectionId: 'gateway',
      headers,
      remoteKind: 'url',
      source: 'settings',
      token: 'secret',
      wsUrl: 'wss://gateway.example.com/api/ws?token=secret'
    },
    [],
    {}
  )

  assert.deepEqual(connection.headers, headers)
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

test('after update clearance, a registry primary is selected before any local spawn', async () => {
  const sshPrimary = { baseUrl: 'http://127.0.0.1:54414', connectionId: '100-106-105-2' }
  let updateCleared = false

  const options = startupOptions({
    resolveRemote: vi.fn(async () => null),
    selectRegistryPrimary: vi.fn(async () => {
      assert.equal(updateCleared, true)

      return sshPrimary
    }),
    waitForLocalStart: vi.fn(async () => {
      updateCleared = true
    }),
    attachHostBackend: vi.fn(async () => ({ baseUrl: 'http://127.0.0.1:49900', mode: 'local' as const }))
  })

  assert.deepEqual(await runPrimaryBackendStartup(options), {
    kind: 'remote',
    connection: { baseUrl: sshPrimary.baseUrl, mode: 'remote' }
  })
  assert.equal(options.prepareLocalBackend.mock.calls.length, 0)
  assert.equal(options.ensureLocalRuntime.mock.calls.length, 0)
  assert.equal(options.attachHostBackend.mock.calls.length, 0)
})

test('a null post-update primary still spawns local exactly once', async () => {
  const options = startupOptions({
    resolveRemote: vi.fn(async () => null),
    selectRegistryPrimary: vi.fn(async () => null)
  })

  assert.equal((await runPrimaryBackendStartup(options)).kind, 'local')
  assert.equal(options.selectRegistryPrimary.mock.calls.length, 1)
  assert.equal(options.ensureLocalRuntime.mock.calls.length, 1)
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

test('invalidating a pending runtime discovery prevents setup and bootstrap', async () => {
  const state = createBackendConnectionState()
  const attempt = state.startAttempt()
  let finish!: (backend: typeof bootstrapBackend) => void
  let started!: () => void

  const resolving = new Promise<void>(resolve => {
    started = resolve
  })

  const options = startupOptions({
    assertCurrentAttempt: () => state.assertCurrentAttempt(attempt),
    prepareLocalBackend: async () => {
      started()

      return new Promise<typeof bootstrapBackend>(resolve => {
        finish = resolve
      })
    }
  })

  const pending = runPrimaryBackendStartup(options)
  await resolving
  state.invalidate()
  finish(bootstrapBackend)
  await assert.rejects(pending, /superseded/)
  assert.equal(options.waitForDecision.mock.calls.length, 0)
  assert.equal(options.ensureLocalRuntime.mock.calls.length, 0)
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
