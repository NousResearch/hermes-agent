import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { registerManagedRolloutDesktopRuntime } from './managed-rollout-desktop-runtime'
import { ManagedConnectionUpdateGate } from './managed-ssh-update'
import { createManagedSshUpdateService } from './managed-ssh-update-service'

test('production composition registers only for the owner and shares legacy admission', async () => {
  const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-rollout-composition-'))
  const userData = path.join(fixtureRoot, 'user-data')
  const gate = new ManagedConnectionUpdateGate()
  const activeUpdates = new Map<string, Promise<any>>()
  const activeRecoveries = new Map<string, Promise<void>>()
  let remoteEffects = 0
  let owned = false
  const sender = {}
  const window = { webContents: sender, isDestroyed: () => false }
  const handlers = new Map<string, (event: { sender: unknown }, payload: unknown) => Promise<unknown>>()

  const service = createManagedSshUpdateService({
    gate,
    activeUpdates,
    activeRecoveries,
    resolveSource: () => null,
    readRecoveryRecords: () => [],
    captureScopes: async () => [],
    openTransport: async () => {remoteEffects += 1; throw new Error('unexpected-remote-transport')},
    targetFromState: () => {throw new Error('unexpected-remote-target')},
    executeRemoteUpdate: async () => {remoteEffects += 1; throw new Error('unexpected-remote-update')},
    preflightRemote: async () => {},
    awaitRestoreClearance: async () => {},
    drainScope: async () => {},
    closeTransports: async () => {},
    restoreScope: async () => {},
    prepareRecovery: async () => {},
    completeRecovery: async () => {},
    restoreRecoveryScope: async () => {}
  })

  const connections = {
    managedConnectionUpdateGate: gate,
    managedConnectionUpdates: activeUpdates,
    managedConnectionRecoveries: activeRecoveries,
    readDesktopConnectionsRegistry: () => ({ connections: [] }),
    effectiveSshConfigFingerprint: async () => 'unused'
  }

  const lifecycle = {
    managedSshUpdateService: service,
    managedSshConfig: () => null,
    openManagedSshUpdateTransport: async () => {remoteEffects += 1; throw new Error('unexpected-remote-transport')},
    captureManagedSshScopes: async () => []
  }

  const deps = {
    app: { getPath: () => userData },
    ipcMain: { handle: (channel: string, handler: (event: { sender: unknown }, payload: unknown) => Promise<unknown>) => {
      assert.equal(handlers.has(channel), false)
      handlers.set(channel, handler)
    } },
    connections,
    lifecycle,
    getMainWindow: () => window,
    processOwner: () => owned
  }

  try {
    assert.throws(() => registerManagedRolloutDesktopRuntime(deps as any), /owner-unavailable/)
    assert.equal(handlers.size, 0)
    assert.equal(fs.existsSync(userData), false)

    owned = true

    const wrongGate = { ...service, gate: new ManagedConnectionUpdateGate() }

    assert.throws(() => registerManagedRolloutDesktopRuntime({
      ...deps, lifecycle: { ...lifecycle, managedSshUpdateService: wrongGate }
    } as any), /shared-update-admission-unavailable/)
    assert.equal(handlers.size, 0)
    assert.equal(fs.existsSync(userData), false)

    registerManagedRolloutDesktopRuntime(deps as any)
    assert.equal(handlers.size, 11)
    assert.equal(service.gate, gate)

    const capabilities = handlers.get('hermes:managed-rollouts:capabilities')!
    const trusted = await capabilities({ sender }, undefined) as any
    assert.equal(trusted.ok, true)
    assert.equal(trusted.value.available, false)
    assert.equal(trusted.value.maxConcurrency, 0)
    assert.equal(trusted.value.maxInstallations, 0)
    const inventory = await handlers.get('hermes:managed-rollouts:inventory')!({ sender }, undefined) as any
    assert.equal(inventory.ok, true)
    assert.deepEqual(inventory.value.observations, [])
    assert.deepEqual(await capabilities({ sender: {} }, undefined), {
      ok: false, code: 'forbidden', message: 'Managed rollout IPC requires a trusted sender.'
    })

    owned = false
    assert.equal((await capabilities({ sender }, undefined) as any).code, 'forbidden')
    assert.equal(remoteEffects, 0)
  } finally {
    assert.equal(path.dirname(path.resolve(fixtureRoot)), path.resolve(os.tmpdir()))
    fs.rmSync(fixtureRoot, { recursive: true, force: true })
  }
})
