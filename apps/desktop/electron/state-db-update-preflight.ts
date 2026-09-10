import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import type { PreUpdateBackupPolicy } from './pre-update-backup-policy'
import {
  createVerifiedSqliteBackup,
  inspectSqliteFootprint,
  type SqliteBackupResult,
  type SqliteFootprint
} from './sqlite-backup'

const SQLITE_HEADER = Buffer.from('SQLite format 3\0')
const MIN_SQLITE_BYTES = 101
const GIB = 1024n * 1024n * 1024n
const MAX_BACKUP_DEADLINE_MS = 30 * 60_000
const MIN_BACKUP_DEADLINE_MS = 2 * 60_000

export type StateDbHeaderInspection = { status: 'missing' } | { headerHex: string; size: number; status: 'valid' }

export interface StateDbPreflightDeps {
  availableBytes: (directory: string) => bigint
  createBackup: (
    sourcePath: string,
    destinationPath: string,
    options: { deadlineMs: number; integrityCheckMaxBytes: number; verificationDeadlineMs: number }
  ) => Promise<SqliteBackupResult>
  inspectFootprint: (sourcePath: string) => SqliteFootprint
  inspectHeader: (sourcePath: string) => StateDbHeaderInspection
  listFiles: (directory: string) => string[]
  nonce: () => string
  now: () => Date
  removeFile: (filePath: string) => void
}

export interface StateDbPreflightOptions {
  hermesHome: string
  policy: PreUpdateBackupPolicy | (() => Promise<PreUpdateBackupPolicy>)
  rememberLog: (message: string) => void
}

export type StateDbPreflightResult =
  | { status: 'missing' }
  | { reason: 'config-disabled' | 'quick-size-cap'; status: 'skipped' }
  | { backup: SqliteBackupResult; path: string; status: 'backed-up' }

export function inspectStateDbHeader(sourcePath: string): StateDbHeaderInspection {
  let stat: fs.Stats

  try {
    stat = fs.statSync(sourcePath)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return { status: 'missing' }
    }

    throw error
  }

  if (!stat.isFile()) {
    throw new Error('state.db is not a regular file')
  }

  if (stat.size < MIN_SQLITE_BYTES) {
    throw new Error(`state.db is too small (${stat.size} bytes) for a valid SQLite database`)
  }

  const descriptor = fs.openSync(sourcePath, 'r')
  const header = Buffer.alloc(SQLITE_HEADER.length)
  let bytesRead = 0

  try {
    bytesRead = fs.readSync(descriptor, header, 0, header.length, 0)
  } finally {
    fs.closeSync(descriptor)
  }

  if (bytesRead !== SQLITE_HEADER.length) {
    throw new Error(`state.db header read was short (${bytesRead}/${SQLITE_HEADER.length} bytes)`)
  }

  if (!header.equals(SQLITE_HEADER)) {
    throw new Error(`state.db has an invalid SQLite header (${header.toString('hex')})`)
  }

  return { headerHex: header.toString('hex'), size: stat.size, status: 'valid' }
}

export function planEmergencyStateDbBackup(
  policy: PreUpdateBackupPolicy,
  estimatedBackupBytes: number
): { reason: 'config-disabled' | 'quick-size-cap'; status: 'skipped' } | { keep: number; status: 'backup' } {
  if (policy.mode === 'off') {
    return { reason: 'config-disabled', status: 'skipped' }
  }

  if (policy.mode === 'quick' && estimatedBackupBytes > policy.quickMaxFileSize) {
    return { reason: 'quick-size-cap', status: 'skipped' }
  }

  return { keep: policy.mode === 'quick' ? policy.quickKeep : policy.backupKeep, status: 'backup' }
}

export function requiredFreeBytes(estimatedBackupBytes: bigint): bigint {
  const proportionalHeadroom = estimatedBackupBytes / 10n
  const headroom = proportionalHeadroom > GIB ? proportionalHeadroom : GIB

  return estimatedBackupBytes + headroom
}

export function backupDeadlineMs(estimatedBackupBytes: number): number {
  const gibibytes = Math.max(1, Math.ceil(estimatedBackupBytes / Number(GIB)))

  return Math.min(MAX_BACKUP_DEADLINE_MS, Math.max(MIN_BACKUP_DEADLINE_MS, gibibytes * 60_000))
}

export function selectEmergencyBackupPrune(files: string[], current: string, keep: number): string[] {
  const previous = files
    .filter(file => file !== current && file.startsWith('state.db.pre-update-emergency-') && file.endsWith('.bak'))
    .sort()
    .reverse()

  return previous.slice(Math.max(0, keep - 1))
}

function defaultAvailableBytes(directory: string): bigint {
  const stats = fs.statfsSync(directory, { bigint: true })

  return stats.bavail * stats.bsize
}

const DEFAULT_DEPS: StateDbPreflightDeps = {
  availableBytes: defaultAvailableBytes,
  createBackup: createVerifiedSqliteBackup,
  inspectFootprint: inspectSqliteFootprint,
  inspectHeader: inspectStateDbHeader,
  listFiles: directory => fs.readdirSync(directory),
  nonce: () => crypto.randomUUID(),
  now: () => new Date(),
  removeFile: filePath => fs.rmSync(filePath, { force: true })
}

function emergencyBackupName(now: Date, nonce: string): string {
  const timestamp = now.toISOString().replace(/[:.]/g, '-')

  return `state.db.pre-update-emergency-${timestamp}-${nonce}.bak`
}

export async function runStateDbUpdatePreflight(
  options: StateDbPreflightOptions,
  deps: StateDbPreflightDeps = DEFAULT_DEPS
): Promise<StateDbPreflightResult> {
  const stateDbPath = path.join(options.hermesHome, 'state.db')
  const header = deps.inspectHeader(stateDbPath)

  if (header.status === 'missing') {
    options.rememberLog('[updates] state.db pre-flight: not found (fresh install?)')

    return { status: 'missing' }
  }

  options.rememberLog(
    `[updates] state.db pre-flight: size=${header.size}, headerOk=true, headerHex=${header.headerHex}`
  )

  const policy = typeof options.policy === 'function' ? await options.policy() : options.policy

  if (policy.mode === 'off') {
    options.rememberLog('[updates] emergency state.db backup disabled by updates.pre_update_backup')

    return { reason: 'config-disabled', status: 'skipped' }
  }

  const footprint = deps.inspectFootprint(stateDbPath)
  const plan = planEmergencyStateDbBackup(policy, footprint.estimatedBackupBytes)

  if (plan.status === 'skipped') {
    options.rememberLog(
      `[updates] emergency state.db backup skipped: estimated ${footprint.estimatedBackupBytes} bytes exceeds ` +
        `quick-mode cap ${policy.quickMaxFileSize}`
    )

    return plan
  }

  const estimatedBytes = BigInt(footprint.estimatedBackupBytes)
  const requiredBytes = requiredFreeBytes(estimatedBytes)
  let availableBytes: bigint

  try {
    availableBytes = deps.availableBytes(options.hermesHome)
  } catch (error) {
    throw new Error('could not measure free space for the emergency state.db backup', { cause: error })
  }

  if (availableBytes < requiredBytes) {
    throw new Error(
      `insufficient free space for the emergency state.db backup: ` +
        `available=${availableBytes} required=${requiredBytes} estimated_snapshot=${estimatedBytes}`
    )
  }

  const filesBefore = deps.listFiles(options.hermesHome)

  for (const stalePartial of filesBefore.filter(
    file => file.startsWith('state.db.pre-update-emergency-') && file.endsWith('.bak.partial')
  )) {
    deps.removeFile(path.join(options.hermesHome, stalePartial))
  }

  const fileName = emergencyBackupName(deps.now(), deps.nonce())
  const destinationPath = path.join(options.hermesHome, fileName)
  const deadlineMs = backupDeadlineMs(footprint.estimatedBackupBytes)

  const backup = await deps.createBackup(stateDbPath, destinationPath, {
    deadlineMs,
    integrityCheckMaxBytes: policy.quickMaxFileSize,
    verificationDeadlineMs: deadlineMs
  })

  options.rememberLog(
    `[updates] emergency state.db backup: ${destinationPath} (${backup.size} bytes, ` +
      `${backup.verification}, ${backup.durationMs}ms)`
  )

  const filesAfter = deps.listFiles(options.hermesHome)

  for (const old of selectEmergencyBackupPrune(filesAfter, fileName, plan.keep)) {
    try {
      deps.removeFile(path.join(options.hermesHome, old))
    } catch (error) {
      options.rememberLog(`[updates] could not prune old emergency state.db backup ${old}: ${String(error)}`)
    }
  }

  return { backup, path: destinationPath, status: 'backed-up' }
}
