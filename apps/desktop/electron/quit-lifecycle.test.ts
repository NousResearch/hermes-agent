import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createQuitLifecycle, type QuitLifecycleDeps } from './quit-lifecycle'

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

function makeHarness(overrides: Partial<QuitLifecycleDeps> = {}) {
  let onTimeout: (() => void) | undefined
  let appQuitting = false
  let handoff = false
  let backendStarted = false
  let managedOps: Promise<unknown>[] = []
  const logs: string[] = []
  const requestQuit = vi.fn()
  const hardExit = vi.fn()
  const flushLogsSync = vi.fn()
  const runWillQuitCleanup = vi.fn()
  const onBeforeQuitCleanup = vi.fn()
  const sealSshBootstrap = vi.fn()
  const onQuitAdmitted = vi.fn()
  const cancelFinalization = vi.fn()

  const scheduleFinalization = vi.fn((callback: () => void) => {
    onTimeout = callback

    return 'finalization-timer'
  })

  const lifecycle = createQuitLifecycle({
    platform: 'darwin',
    skipQuitConfirm: false,
    isQuittingForHandoff: () => handoff,
    hasBackendShutdownStarted: () => backendStarted,
    getActiveWorkReports: () => [],
    quitStopsBackendWork: () => true,
    restoreTrayBeforePrompt: () => {},
    showQuitPrompt: () => null,
    setAppQuitting: quitting => {
      appQuitting = quitting
    },
    onQuitAdmitted,
    getManagedUpdateOperations: () => managedOps,
    sealSshBootstrap,
    getBackendActivity: () => ({ connectionPending: false, processAttached: false, shutdownPending: false }),
    hasSshTeardownWork: () => false,
    runBackendShutdown: async () => {
      backendStarted = true
    },
    runSshTeardown: async () => {},
    onBeforeQuitCleanup,
    runWillQuitCleanup,
    log: message => logs.push(message),
    flushLogsSync,
    requestQuit,
    hardExit,
    scheduleFinalization,
    cancelFinalization,
    ...overrides
  })

  return {
    lifecycle,
    logs,
    requestQuit,
    hardExit,
    flushLogsSync,
    runWillQuitCleanup,
    onBeforeQuitCleanup,
    sealSshBootstrap,
    onQuitAdmitted,
    scheduleFinalization,
    cancelFinalization,
    getAppQuitting: () => appQuitting,
    setHandoff: (value: boolean) => {
      handoff = value
    },
    setManagedOps: (ops: Promise<unknown>[]) => {
      managedOps = ops
    },
    triggerFinalizationTimeout: () => onTimeout?.()
  }
}

test('deferred macOS teardown yields to a macrotask, arms finalization, and cancels on quit', async () => {
  const sshDrain = deferred()

  const harness = makeHarness({
    hasSshTeardownWork: () => true,
    runSshTeardown: () => sshDrain.promise
  })

  const firstPrevent = vi.fn()
  harness.lifecycle.handleBeforeQuit({ preventDefault: firstPrevent })
  assert.equal(firstPrevent.mock.calls.length, 1)
  assert.equal(harness.lifecycle.isQuitInProgress(), true)
  assert.equal(harness.lifecycle.isQuitUnderway(), true)
  assert.equal(harness.scheduleFinalization.mock.calls.length, 0)

  // Dock activate while teardown is still pending must not recreate or kick final quit early.
  assert.equal(harness.lifecycle.handleActivateWhileQuitting(), true)
  assert.equal(harness.scheduleFinalization.mock.calls.length, 0)
  assert.equal(harness.requestQuit.mock.calls.length, 0)

  sshDrain.resolve()
  await Promise.resolve()
  await Promise.resolve()
  assert.equal(harness.requestQuit.mock.calls.length, 0)

  await new Promise(resolve => setTimeout(resolve, 0))
  assert.equal(harness.scheduleFinalization.mock.calls.length, 1)
  assert.equal(harness.requestQuit.mock.calls.length, 1)

  const secondPrevent = vi.fn()
  harness.lifecycle.handleBeforeQuit({ preventDefault: secondPrevent })
  assert.equal(secondPrevent.mock.calls.length, 0)

  harness.lifecycle.handleWillQuit()
  harness.lifecycle.handleQuit()
  assert.equal(harness.runWillQuitCleanup.mock.calls.length, 1)
  assert.deepEqual(harness.cancelFinalization.mock.calls, [['finalization-timer']])
  harness.triggerFinalizationTimeout()
  assert.equal(harness.hardExit.mock.calls.length, 0)
})

test('stalled pre-will-quit macOS exit runs idempotent will-quit cleanup and flushes logs before hardExit', async () => {
  const harness = makeHarness()

  const prevent = vi.fn()
  harness.lifecycle.handleBeforeQuit({ preventDefault: prevent })
  assert.equal(prevent.mock.calls.length, 0)
  assert.equal(harness.scheduleFinalization.mock.calls.length, 1)

  // Electron stalls before emitting will-quit; window-all-closed also requests quit.
  harness.lifecycle.handleWindowAllClosed()
  assert.equal(harness.requestQuit.mock.calls.length, 1)

  harness.triggerFinalizationTimeout()
  assert.equal(harness.runWillQuitCleanup.mock.calls.length, 1)
  assert.equal(harness.flushLogsSync.mock.calls.length, 1)
  assert.deepEqual(harness.hardExit.mock.calls, [[0]])
  assert.ok(harness.logs.some(line => line.includes('forcing darwin process exit')))

  // A subsequent will-quit must not run cleanup twice.
  harness.lifecycle.handleWillQuit()
  assert.equal(harness.runWillQuitCleanup.mock.calls.length, 1)
})

test('latches past managed-update wait once teardown starts so a late operation cannot re-defer quit', async () => {
  const sshDrain = deferred()
  const lateUpdate = deferred()

  const harness = makeHarness({
    hasSshTeardownWork: () => true,
    runSshTeardown: () => sshDrain.promise
  })

  const firstPrevent = vi.fn()
  harness.lifecycle.handleBeforeQuit({ preventDefault: firstPrevent })
  assert.equal(firstPrevent.mock.calls.length, 1)

  // A late managed update appears while teardown is draining.
  harness.setManagedOps([lateUpdate.promise])
  sshDrain.resolve()
  await sshDrain.promise
  await Promise.resolve()
  await new Promise(resolve => setTimeout(resolve, 0))
  assert.equal(harness.scheduleFinalization.mock.calls.length, 1)

  const secondPrevent = vi.fn()
  harness.lifecycle.handleBeforeQuit({ preventDefault: secondPrevent })
  assert.equal(secondPrevent.mock.calls.length, 0)
})

test('skips macOS quitFinalization during a Squirrel update handoff', () => {
  const harness = makeHarness()
  harness.setHandoff(true)

  harness.lifecycle.handleBeforeQuit({ preventDefault: vi.fn() })
  harness.lifecycle.handleWillQuit()

  assert.equal(harness.scheduleFinalization.mock.calls.length, 0)
})

test('window-all-closed stays alive on macOS outside quit and active-work guard cannot re-prompt mid-quit', () => {
  const showQuitPrompt = vi.fn(() => Promise.resolve(false))

  const harness = makeHarness({
    getActiveWorkReports: () => [{ count: 1, titles: ['Long task'] }],
    showQuitPrompt
  })

  // Normal macOS last-window close stays alive in the Dock.
  harness.lifecycle.handleWindowAllClosed()
  assert.equal(harness.requestQuit.mock.calls.length, 0)

  // First before-quit prompts for active work and keeps quitInProgress false.
  const firstPrevent = vi.fn()
  harness.lifecycle.handleBeforeQuit({ preventDefault: firstPrevent })
  assert.equal(firstPrevent.mock.calls.length, 1)
  assert.equal(showQuitPrompt.mock.calls.length, 1)
  assert.equal(harness.lifecycle.isQuitInProgress(), false)
  assert.equal(harness.getAppQuitting(), false)
})
