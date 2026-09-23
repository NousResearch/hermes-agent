import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createUpdateGateRuntime } from './update-gate-runtime'

function fixture(overrides: Record<string, unknown> = {}) {
  const events: string[] = []
  let inFlight = true
  let handoffActive = false

  const runtime = createUpdateGateRuntime({
    hermesHome: 'C:/test/hermes',
    isPackaged: false,
    installStamp: null,
    loadInstallStamp: () => null,
    getUpdateInFlight: () => inFlight,
    getHandoffActive: () => handoffActive,
    readLiveUpdateMarker: () => null,
    readAndConsumeHandoffResult: () => null,
    waitForUpdateClearance: async (gate, options) => {
      assert.equal(gate.isUpdateInFlight(), true)
      assert.equal(gate.isHandoffActive(), false)
      assert.equal(options.timeoutMs, 20 * 60 * 1000)

      return 'clear' as const
    },
    detectBundleSwap: () => false,
    buildNoSandboxRelaunchArgs: args => args,
    app: { relaunch: () => events.push('relaunch') },
    dialog: { showMessageBox: async () => ({ response: 2 }) },
    shell: { showItemInFolder: () => events.push('open-log') },
    localBackendLifecycle: { signal: new AbortController().signal, assertCanStart: () => events.push('assert') },
    firstRunBoot: { advanceBootProgress: async () => {} },
    rememberLog: message => events.push(message),
    sendOpenUpdatesRequested: () => events.push('open-updates'),
    exitAfterBackendShutdown: async () => events.push('exit'),
    ...overrides
  })

  return {
    runtime,
    events,
    setInFlight: (value: boolean) => {
      inFlight = value
    },
    setHandoffActive: (value: boolean) => {
      handoffActive = value
    }
  }
}

test('primary and pool gates consult the same live in-flight and handoff state', async () => {
  const { runtime, setInFlight, setHandoffActive } = fixture()

  assert.equal(runtime.updateGateDeps().isUpdateInFlight(), true)
  assert.equal(await runtime.waitForUpdateToFinish(), false)
  setInFlight(false)
  setHandoffActive(true)
  assert.equal(runtime.updateGateDeps().isUpdateInFlight(), false)
  assert.equal(runtime.updateGateDeps().isHandoffActive(), true)
})

test('detached failure receipt is consumed at the gate and opens the requested log', async () => {
  let consumed = 0

  const { runtime, events } = fixture({
    readAndConsumeHandoffResult: () => {
      consumed += 1

      return { ok: false, exitCode: 7, message: 'update failed' }
    },
    dialog: { showMessageBox: async () => ({ response: 1 }) }
  })

  assert.equal(await runtime.waitForUpdateToFinish(), false)
  await Promise.resolve()
  assert.equal(consumed, 1)
  assert.ok(events.some(event => event.includes('detached update FAILED')))
  assert.deepEqual(
    events.filter(event => event === 'open-log'),
    ['open-log']
  )
})
