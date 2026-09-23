import fs from 'node:fs'
import path from 'node:path'

import { backendScopePrefix } from './connection-registry'
import { tightenSecretFileMode, writeSecretFileAtomic } from './hardening'
import { managedSshRecoveryScopes, validateCorrelationId } from './managed-ssh-update'

export function createManagedSshRecoveryJournal(recoveryPath: string) {
  const DESKTOP_MANAGED_SSH_RECOVERY_PATH = recoveryPath

  function readManagedSshRecoveryRecords(): any[] {
    try {
      const stat = fs.lstatSync(DESKTOP_MANAGED_SSH_RECOVERY_PATH)

      if (!stat.isFile() || stat.isSymbolicLink() || !tightenSecretFileMode(DESKTOP_MANAGED_SSH_RECOVERY_PATH)) {
        throw new Error('Managed SSH recovery journal is not a safe owner-only file.')
      }

      const payload = JSON.parse(fs.readFileSync(DESKTOP_MANAGED_SSH_RECOVERY_PATH, 'utf8'))

      if (payload?.version !== 1 || !Array.isArray(payload.records)) {
        throw new Error('Managed SSH recovery journal has an unsupported shape.')
      }

      const valid = payload.records.every(record => {
        if (
          !record ||
          typeof record !== 'object' ||
          typeof record.connectionId !== 'string' ||
          record.source?.kind !== 'ssh' ||
          record.source?.id !== record.connectionId ||
          (record.installationId !== undefined &&
            (typeof record.installationId !== 'string' || !/^[0-9a-f]{32}$/.test(record.installationId))) ||
          !['prepared', 'launching'].includes(record.phase) ||
          !Array.isArray(record.scopes) ||
          record.scopes.length > 256
        ) {
          return false
        }

        try {
          validateCorrelationId(record.correlationId)
        } catch {
          return false
        }

        const scopesValid = record.scopes.every(
          scope =>
            scope &&
            typeof scope === 'object' &&
            typeof scope.key === 'string' &&
            scope.key.length <= 256 &&
            typeof scope.profile === 'string' &&
            scope.profile.length > 0 &&
            scope.profile.length <= 128 &&
            ['legacy', 'primary', 'registry'].includes(scope.kind) &&
            (scope.kind === 'primary' || scope.key.length > 0)
        )

        const identities = record.scopes.map(scope => `${scope.kind}\0${scope.key}\0${scope.profile}`)

        return (
          scopesValid &&
          new Set(identities).size === identities.length &&
          record.scopes.filter(scope => scope.kind === 'primary').length <= 1
        )
      })

      if (!valid) {
        throw new Error('Managed SSH recovery journal contains an invalid record.')
      }

      return payload.records
    } catch (cause: any) {
      if (cause?.code === 'ENOENT') {
        return []
      }

      const error: any = new Error(
        'Managed SSH recovery state is unreadable or malformed; refusing connection startup and edits.'
      )

      error.code = 'managed-update-recovery-unavailable'
      error.cause = cause
      throw error
    }
  }

  function writeManagedSshRecoveryRecords(records) {
    fs.mkdirSync(path.dirname(DESKTOP_MANAGED_SSH_RECOVERY_PATH), { recursive: true })
    writeSecretFileAtomic(
      DESKTOP_MANAGED_SSH_RECOVERY_PATH,
      JSON.stringify({ version: 1, records, updatedAt: new Date().toISOString() }, null, 2)
    )
  }

  function persistManagedSshRecovery(source, correlationId, scopes, installationId?: string) {
    if (installationId !== undefined && !/^[0-9a-f]{32}$/.test(installationId)) {
      throw new Error('Managed SSH recovery installation identity is invalid.')
    }
    const prefix = backendScopePrefix(source.id)
    const recoveryScopes = managedSshRecoveryScopes(scopes, prefix)

    const records = readManagedSshRecoveryRecords().filter(record => record.connectionId !== source.id)
    records.push({
      connectionId: source.id,
      correlationId: validateCorrelationId(correlationId),
      ...(installationId ? { installationId } : {}),
      createdAt: new Date().toISOString(),
      phase: 'prepared',
      scopes: recoveryScopes,
      // Registry secrets are already safeStorage envelopes. Persist the exact
      // connection snapshot so crash recovery does not silently switch hosts or
      // credentials after a Settings edit.
      source
    })
    writeManagedSshRecoveryRecords(records)
  }

  function markManagedSshRecoveryLaunching(connectionId, correlationId) {
    const records = readManagedSshRecoveryRecords()

    const index = records.findIndex(
      record => record.connectionId === connectionId && record.correlationId === correlationId
    )

    if (index < 0) {
      throw new Error('Managed SSH recovery record disappeared before remote update launch.')
    }

    records[index] = { ...records[index], phase: 'launching' }
    writeManagedSshRecoveryRecords(records)
  }

  function clearManagedSshRecovery(connectionId, correlationId) {
    const records = readManagedSshRecoveryRecords()

    const remaining = records.filter(
      record => record.connectionId !== connectionId || record.correlationId !== correlationId
    )

    if (remaining.length === records.length) {
      return
    }

    if (remaining.length > 0) {
      writeManagedSshRecoveryRecords(remaining)
    } else {
      try {
        fs.unlinkSync(DESKTOP_MANAGED_SSH_RECOVERY_PATH)
      } catch (error: any) {
        if (error?.code !== 'ENOENT') {
          throw error
        }
      }
    }
  }

  return {
    readManagedSshRecoveryRecords,
    writeManagedSshRecoveryRecords,
    persistManagedSshRecovery,
    markManagedSshRecoveryLaunching,
    clearManagedSshRecovery
  }
}
