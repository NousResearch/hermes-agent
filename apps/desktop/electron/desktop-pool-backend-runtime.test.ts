import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createDesktopPoolBackendRuntime } from './desktop-pool-backend-runtime'
import { releaseLocalBackendSlotAfterExit } from './pool-spawn-coordinator'
import { createPoolStopper } from './pool-stop'

test('pool stop retains the live child and slot until the shared exit fence settles', async () => {
  const events: string[] = []
  let finishExit!: () => void

  const exited = new Promise<void>(resolve => {
    finishExit = resolve
  })

  const child = { exitCode: null, signalCode: null }

  const entry = {
    process: child,
    releaseLocalBackendSlot: () => events.push('release'),
    localBackendSpawnRequest: null,
    localBackendSlotKey: 'named'
  }

  const backendPool = new Map([['named', entry]])
  let stopperPool: unknown
  let retirerPool: unknown

  const runtime = createDesktopPoolBackendRuntime({
    backendPool,
    createPoolStopper: deps => {
      stopperPool = deps.pool

      return createPoolStopper(deps)
    },
    createPoolRetirer: deps => {
      retirerPool = deps.pool

      return { dispose: () => undefined }
    },
    createPoolRetirementClient: () => ({}),
    releaseLocalBackendSlotAfterExit,
    stopBackendChild: () => events.push('stop'),
    waitForBackendExit: () => exited,
    sshBootstrapCoordinator: { cancelAndWait: async () => events.push('ssh') },
    teardownSshConnection: async () => undefined,
    sshRememberLog: () => undefined,
    rememberLog: () => undefined,
    localBackendSpawnCoordinator: {},
    fetchJson: () => undefined
  } as any)

  assert.equal(stopperPool, backendPool)
  assert.equal(retirerPool, backendPool)
  const stopping = runtime.stopPoolBackend('named')
  const duplicate = runtime.stopPoolBackend('named')

  assert.equal(backendPool.has('named'), false)
  assert.deepEqual(events, ['stop'])
  assert.ok(runtime.poolStopper.inFlight('named'))

  finishExit()
  await Promise.all([stopping, duplicate])
  assert.deepEqual(events, ['stop', 'ssh', 'release'])
  assert.equal(entry.releaseLocalBackendSlot, null)
})
