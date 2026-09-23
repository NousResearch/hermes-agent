import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { createManagedSshRecoveryJournal } from './managed-ssh-recovery-journal'
import { ManagedConnectionUpdateGate } from './managed-ssh-update'

test('a managed SSH recovery journal keeps the connection gated across runtime recreation', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'hermes-ssh-recovery-'))
  const journalPath = path.join(directory, 'recovery.json')
  const connectionId = 'homelab'
  const correlationId = '12345678-1234-4678-9234-567812345678'
  const source = { id: connectionId, kind: 'ssh', label: 'Homelab' }

  try {
    const first = createManagedSshRecoveryJournal(journalPath)
    first.persistManagedSshRecovery(source, correlationId, [
      { key: 'conn:homelab::default', profile: 'default', registryScoped: true }
    ])

    const restored = createManagedSshRecoveryJournal(journalPath)

    const gate = new ManagedConnectionUpdateGate(
      id => restored.readManagedSshRecoveryRecords().find(record => record.connectionId === id)?.correlationId || null
    )

    assert.equal(restored.readManagedSshRecoveryRecords()[0].phase, 'prepared')
    assert.throws(() => gate.assertCanDial(connectionId), /paused/)
    restored.markManagedSshRecoveryLaunching(connectionId, correlationId)
    assert.equal(first.readManagedSshRecoveryRecords()[0].phase, 'launching')
    restored.clearManagedSshRecovery(connectionId, correlationId)
    assert.deepEqual(first.readManagedSshRecoveryRecords(), [])
    assert.doesNotThrow(() => gate.assertCanDial(connectionId))
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('a recovery record retains the canonical installation fence across runtime recreation', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'hermes-ssh-recovery-'))
  const journalPath = path.join(directory, 'recovery.json')
  const correlationId = '12345678-1234-4678-9234-567812345678'
  const installationId = 'a'.repeat(32)
  const source = { id: 'alias-a', kind: 'ssh', label: 'Alias A' }

  try {
    createManagedSshRecoveryJournal(journalPath).persistManagedSshRecovery(
      source, correlationId, [], installationId
    )

    assert.deepEqual(
      createManagedSshRecoveryJournal(journalPath).readManagedSshRecoveryRecords().map(record => ({
        connectionId: record.connectionId,
        installationId: record.installationId
      })),
      [{ connectionId: source.id, installationId }]
    )
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
