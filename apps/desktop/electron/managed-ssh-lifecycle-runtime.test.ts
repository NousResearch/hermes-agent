import assert from 'node:assert/strict'

import { test } from 'vitest'

import { normalizeSshConfig } from './connection-config'
import { createManagedSshLifecycleRuntime } from './managed-ssh-lifecycle-runtime'
import { ManagedConnectionUpdateGate } from './managed-ssh-update'

test('a restored managed SSH profile keeps the captured host and maps its remote profile', () => {
  const gate = new ManagedConnectionUpdateGate()

  const runtime = createManagedSshLifecycleRuntime({
    managedConnectionUpdateGate: gate,
    normalizeSshConfig
  })

  const source = {
    id: 'homelab',
    kind: 'ssh',
    host: 'mini.example',
    user: 'axl',
    port: 22,
    remoteProfile: '',
    keyPath: ''
  }

  assert.deepEqual(runtime.managedSshConfig(source, 'mara'), {
    mode: 'ssh',
    host: 'mini.example',
    user: 'axl',
    remoteProfile: 'mara'
  })
  assert.equal(runtime.managedConnectionUpdateGate, gate)
})

const RECOVERY_CORRELATION = '12345678-1234-4678-9234-567812345678'
const RECORDED_INSTALL_ID = 'a'.repeat(32)
const FOREIGN_INSTALL_ID = 'd'.repeat(32)

function lifecycleFor(observedInstallId: string, records: any[]) {
  const gate = new ManagedConnectionUpdateGate()
  const cleared: string[] = []
  const restored: unknown[] = []

  const ssh = {
    open: async () => undefined,
    close: async () => undefined,
    exec: async (command: string) => {
      if (command.includes('echo "${HERMES_HOME:-$HOME/.hermes}"')) {return '~/.hermes'}

      if (command.includes('if [ -f')) {return observedInstallId}

      return ''
    }
  }

  const runtime = createManagedSshLifecycleRuntime({
    managedConnectionUpdateGate: gate,
    managedConnectionRecoveries: new Map(),
    managedConnectionUpdates: new Map(),
    managedPrimaryRestoreOwners: new Map(),
    backendPool: new Map(),
    normalizeSshConfig,
    createSshProbeConnection: () => ssh,
    detectRemotePlatform: async () => ({ os: 'Linux' }),
    remoteLifecycle: {
      locateHermes: async () => '/srv/hermes-agent/.venv/bin/hermes',
      probeRemoteHermesHome: async () => '~/.hermes'
    },
    readManagedSshRecoveryRecords: () => records,
    clearManagedSshRecovery: (connectionId: string, correlationId: string) => {
      cleared.push(`${connectionId}:${correlationId}`)
    },
    waitForManagedRemoteClearance: async () => undefined,
    restoreRecoveryScope: async (record: unknown) => { restored.push(record) },
    sshRememberLog: () => undefined,
    backendScopeKey: (connectionId: string, profile: string) => `${connectionId}:${profile}`
  })

  return { runtime, gate, cleared, restored }
}

function recoveryRecord(installationId: string | undefined) {
  return {
    connectionId: 'homelab',
    correlationId: RECOVERY_CORRELATION,
    ...(installationId === undefined ? {} : { installationId }),
    phase: 'launching',
    scopes: [],
    source: { id: 'homelab', kind: 'ssh', label: 'homelab', host: 'mini.example', user: 'axl', port: 22 }
  }
}

test('startup recovery clears the durable record only when the selected remote re-proves its installation', async () => {
  const records = [recoveryRecord(RECORDED_INSTALL_ID)]
  const { runtime, cleared } = lifecycleFor(RECORDED_INSTALL_ID, records)

  await runtime.resumeManagedSshRecoveries()

  assert.deepEqual(cleared, [`homelab:${RECOVERY_CORRELATION}`])
})

test('startup recovery refuses to touch a record whose selected remote is a different installation', async () => {
  const records = [recoveryRecord(RECORDED_INSTALL_ID)]
  const { runtime, cleared } = lifecycleFor(FOREIGN_INSTALL_ID, records)

  await runtime.resumeManagedSshRecoveries()

  assert.deepEqual(cleared, [])
})

test('startup recovery keeps the obligation pending when the selected remote cannot prove any installation', async () => {
  const records = [recoveryRecord(RECORDED_INSTALL_ID)]
  const { runtime, cleared } = lifecycleFor('', records)

  await runtime.resumeManagedSshRecoveries()

  assert.deepEqual(cleared, [])
})
