import assert from 'node:assert/strict'

import { test } from 'vitest'

import { registerDesktopBootstrapIpc } from './desktop-bootstrap-ipc'

test('bootstrap reset and repair clear latches while preserving the repair decision', async () => {
  const handlers = new Map<string, (...args: any[]) => Promise<any>>()
  const calls: string[] = []
  let repairAttempt = 0
  let repairRequested = false
  let latch = 'failed'
  const ipcMain = { handle: (name: string, fn: (...args: any[]) => Promise<any>) => handlers.set(name, fn) }

  registerDesktopBootstrapIpc({
    ipcMain,
    recycleOwnedBackend: async () => {},
    sendConnectionApplied: () => {},
    primaryProfileKey: () => 'main',
    teardownPoolBackendAndWait: async () => {},
    teardownPrimaryBackendAndWait: async () => { calls.push('teardown') },
    teardownSshConnection: async () => {},
    rememberLog: () => {},
    clearFailures: () => { latch = '' },
    firstRunBoot: {
      getFirstRunSetupGate: () => ({
        resetForRetry: () => calls.push('retry'),
        resetForRepair: () => calls.push('repair')
      }),
      resetBootstrapSnapshot: () => calls.push('snapshot'),
      getBootProgressState: () => ({ step: 'ready' }),
      getBootstrapState: () => ({ ready: true }),
      continueFirstRunLocalBootstrap: () => {}
    },
    incrementRepairAttempt: () => ++repairAttempt,
    maxBootstrapRepairSoftAttempts: 3,
    getPrimaryBackendProcess: () => ({ exitCode: null, signalCode: null }),
    setBootstrapRepairRequested: value => { repairRequested = value },
    resetHermesConnection: () => calls.push('reset-connection'),
    getBootstrapAbortController: () => null
  } as any)

  assert.deepEqual(await handlers.get('hermes:bootstrap:reset')!(), { ok: true })
  assert.deepEqual(calls, ['teardown', 'retry', 'snapshot'])
  assert.equal(latch, '')

  latch = 'failed-again'
  assert.deepEqual(await handlers.get('hermes:bootstrap:repair')!(), { ok: true })
  assert.equal(repairAttempt, 1)
  assert.equal(repairRequested, false)
  assert.equal(latch, '')
  assert.deepEqual(calls.slice(3), ['repair', 'reset-connection'])
})

test('bootstrap cancel reports the actual abort outcome and registers getters', async () => {
  const handlers = new Map<string, (...args: any[]) => Promise<any>>()
  const controller = new AbortController()
  registerDesktopBootstrapIpc({
    ipcMain: { handle: (name: string, fn: (...args: any[]) => Promise<any>) => handlers.set(name, fn) },
    getBootstrapAbortController: () => controller,
    firstRunBoot: {
      getBootProgressState: () => ({ step: 'installing' }),
      getBootstrapState: () => ({ ready: false })
    }
  } as any)

  assert.deepEqual(await handlers.get('hermes:bootstrap:cancel')!(), { ok: true, cancelled: true })
  assert.equal(controller.signal.aborted, true)
  assert.deepEqual(await handlers.get('hermes:boot-progress:get')!(), { step: 'installing' })
  assert.deepEqual(await handlers.get('hermes:bootstrap:get')!(), { ready: false })
})
