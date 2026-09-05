import fs from 'node:fs'
import { DatabaseSync } from 'node:sqlite'
import { Worker } from 'node:worker_threads'

const DEFAULT_BACKUP_DEADLINE_MS = 30_000
const DEFAULT_VERIFICATION_DEADLINE_MS = 30_000
const DEFAULT_INTEGRITY_CHECK_MAX_BYTES = 2 * 1024 * 1024 * 1024
const BACKUP_RATE_PAGES = 100

const BACKUP_WORKER_SOURCE = `
  const { parentPort, workerData } = require('node:worker_threads')
  const { backup, DatabaseSync } = require('node:sqlite')
  let source
  ;(async () => {
    try {
      source = new DatabaseSync(workerData.sourcePath)
      source.exec('PRAGMA busy_timeout = 1000')
      source.prepare('SELECT count(*) FROM sqlite_master').get()
      await backup(source, workerData.partialPath, { rate: workerData.rate })
      parentPort.postMessage({ ok: true })
    } catch (error) {
      parentPort.postMessage({ error: error instanceof Error ? error.message : String(error), ok: false })
    } finally {
      try { source?.close() } catch {}
    }
  })()
`

const VERIFY_WORKER_SOURCE = `
  const { parentPort, workerData } = require('node:worker_threads')
  const { DatabaseSync } = require('node:sqlite')
  let snapshot
  try {
    snapshot = new DatabaseSync(workerData.snapshotPath, { readOnly: true })
    if (workerData.verification === 'schema-probe') {
      snapshot.prepare('PRAGMA schema_version').get()
      snapshot.prepare('SELECT count(*) FROM sqlite_master').get()
    } else {
      const rows = snapshot.prepare('PRAGMA integrity_check').all()
      if (rows.length !== 1 || Object.values(rows[0])[0] !== 'ok') {
        throw new Error('SQLite integrity_check failed for emergency backup')
      }
    }
    parentPort.postMessage({ ok: true })
  } catch (error) {
    parentPort.postMessage({ error: error instanceof Error ? error.message : String(error), ok: false })
  } finally {
    try { snapshot?.close() } catch {}
  }
`

export type SqliteBackupPhase = 'backup-complete' | 'published' | 'verified'

export interface SqliteBackupResult {
  durationMs: number
  size: number
  verification: 'integrity-check' | 'schema-probe'
}

export interface SqliteFootprint {
  estimatedBackupBytes: number
  logicalBytes: number
  mainBytes: number
  walBytes: number
}

interface SqliteBackupOptions {
  deadlineMs?: number
  integrityCheckMaxBytes?: number
  onPhase?: (phase: SqliteBackupPhase) => void
  verificationDeadlineMs?: number
}

export function sqliteVerificationMode(
  size: number,
  integrityCheckMaxBytes = DEFAULT_INTEGRITY_CHECK_MAX_BYTES
): SqliteBackupResult['verification'] {
  return integrityCheckMaxBytes > 0 && size > integrityCheckMaxBytes ? 'schema-probe' : 'integrity-check'
}

function fileSizeOrZero(filePath: string): number {
  try {
    return fs.statSync(filePath).size
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {return 0}
    throw error
  }
}

/** Conservative destination estimate including logical pages and WAL bytes. */
export function inspectSqliteFootprint(sourcePath: string): SqliteFootprint {
  const mainBytes = fs.statSync(sourcePath).size
  const walBytes = fileSizeOrZero(`${sourcePath}-wal`)
  const source = new DatabaseSync(sourcePath, { readOnly: true })

  try {
    source.exec('PRAGMA busy_timeout = 1000')
    const pageCountRow = source.prepare('PRAGMA page_count').get() as Record<string, number | bigint>
    const pageSizeRow = source.prepare('PRAGMA page_size').get() as Record<string, number | bigint>
    const pageCount = Number(Object.values(pageCountRow)[0])
    const pageSize = Number(Object.values(pageSizeRow)[0])
    const logicalBytes = pageCount * pageSize

    if (!Number.isSafeInteger(logicalBytes) || logicalBytes <= 0) {
      throw new Error(`SQLite reported an invalid logical size (${logicalBytes})`)
    }

    return {
      estimatedBackupBytes: Math.max(logicalBytes, mainBytes + walBytes),
      logicalBytes,
      mainBytes,
      walBytes
    }
  } finally {
    source.close()
  }
}

export async function createVerifiedSqliteBackup(
  sourcePath: string,
  destinationPath: string,
  options: SqliteBackupOptions = {}
): Promise<SqliteBackupResult> {
  const startedAt = Date.now()
  const deadlineMs = options.deadlineMs ?? DEFAULT_BACKUP_DEADLINE_MS
  const partialPath = `${destinationPath}.partial`

  fs.statSync(sourcePath)
  assertDestinationAbsent(destinationPath)
  removeFile(partialPath)

  try {
    await runBoundedWorker(
      BACKUP_WORKER_SOURCE,
      { partialPath, rate: BACKUP_RATE_PAGES, sourcePath },
      deadlineMs,
      'SQLite emergency backup'
    )
    options.onPhase?.('backup-complete')

    const size = fs.statSync(partialPath).size

    if (size <= 100) {throw new Error(`SQLite emergency backup is too small (${size} bytes)`)}

    const verification = sqliteVerificationMode(
      size,
      options.integrityCheckMaxBytes ?? DEFAULT_INTEGRITY_CHECK_MAX_BYTES
    )

    await runBoundedWorker(
      VERIFY_WORKER_SOURCE,
      { snapshotPath: partialPath, verification },
      options.verificationDeadlineMs ?? Math.max(DEFAULT_VERIFICATION_DEADLINE_MS, deadlineMs),
      'SQLite emergency backup verification'
    )
    options.onPhase?.('verified')

    assertDestinationAbsent(destinationPath)
    // Hard-link publication is atomic and no-replace on Windows and POSIX.
    fs.linkSync(partialPath, destinationPath)
    fs.unlinkSync(partialPath)
    options.onPhase?.('published')

    return { durationMs: Date.now() - startedAt, size, verification }
  } catch (error) {
    removeFile(partialPath)
    throw error
  }
}

function assertDestinationAbsent(destinationPath: string): void {
  if (fs.existsSync(destinationPath)) {
    throw new Error(`SQLite emergency backup destination already exists: ${destinationPath}`)
  }
}

async function runBoundedWorker(
  source: string,
  workerData: Record<string, unknown>,
  timeoutMs: number,
  label: string
): Promise<void> {
  const worker = new Worker(source, { eval: true, workerData })
  let timer: NodeJS.Timeout | undefined

  try {
    await new Promise<void>((resolve, reject) => {
      let settled = false

      const finish = (error?: Error) => {
        if (settled) {return}
        settled = true

        if (error) {reject(error)}
        else {resolve()}
      }

      timer = setTimeout(() => finish(new Error(`${label} exceeded ${timeoutMs}ms deadline`)), timeoutMs)
      timer.unref?.()
      worker.once('message', message => {
        if (message?.ok) {finish()}
        else {finish(new Error(message?.error || `${label} failed`))}
      })
      worker.once('error', error => finish(error))
      worker.once('exit', code => {
        if (!settled) {finish(new Error(`${label} worker exited ${code} before reporting completion`))}
      })
    })
  } finally {
    if (timer) {clearTimeout(timer)}
    await worker.terminate()
  }
}

function removeFile(filePath: string): void {
  try {
    fs.unlinkSync(filePath)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {throw error}
  }
}
