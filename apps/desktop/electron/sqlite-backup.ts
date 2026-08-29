import fs from 'node:fs'
import { backup, DatabaseSync } from 'node:sqlite'

const DEFAULT_BACKUP_DEADLINE_MS = 30_000
const DEFAULT_INTEGRITY_CHECK_MAX_BYTES = 2 * 1024 * 1024 * 1024
const BACKUP_RATE_PAGES = 100

export interface SqliteBackupResult {
  durationMs: number
  size: number
  verification: 'integrity-check' | 'schema-probe'
}

interface SqliteBackupOptions {
  deadlineMs?: number
  integrityCheckMaxBytes?: number
}

export function sqliteVerificationMode(
  size: number,
  integrityCheckMaxBytes = DEFAULT_INTEGRITY_CHECK_MAX_BYTES
): SqliteBackupResult['verification'] {
  return integrityCheckMaxBytes > 0 && size > integrityCheckMaxBytes ? 'schema-probe' : 'integrity-check'
}

export async function createVerifiedSqliteBackup(
  sourcePath: string,
  destinationPath: string,
  options: SqliteBackupOptions = {}
): Promise<SqliteBackupResult> {
  const startedAt = Date.now()
  const deadlineMs = options.deadlineMs ?? DEFAULT_BACKUP_DEADLINE_MS
  const partialPath = `${destinationPath}.partial`
  let source: DatabaseSync | null = null

  fs.statSync(sourcePath)
  removeFile(partialPath)

  try {
    source = new DatabaseSync(sourcePath)
    source.exec('PRAGMA busy_timeout = 1000')
    source.prepare('SELECT count(*) FROM sqlite_master').get()

    await backup(source, partialPath, {
      rate: BACKUP_RATE_PAGES,
      progress: () => {
        if (Date.now() - startedAt >= deadlineMs) {
          throw new Error(`SQLite emergency backup exceeded ${deadlineMs}ms deadline`)
        }
      }
    })

    const size = fs.statSync(partialPath).size

    if (size <= 100) {
      throw new Error(`SQLite emergency backup is too small (${size} bytes)`)
    }

    const verification = verifySqliteSnapshot(
      partialPath,
      size,
      options.integrityCheckMaxBytes ?? DEFAULT_INTEGRITY_CHECK_MAX_BYTES
    )

    fs.renameSync(partialPath, destinationPath)

    return { durationMs: Date.now() - startedAt, size, verification }
  } catch (error) {
    removeFile(partialPath)
    throw error
  } finally {
    source?.close()
  }
}

function verifySqliteSnapshot(
  snapshotPath: string,
  size: number,
  integrityCheckMaxBytes: number
): SqliteBackupResult['verification'] {
  const snapshot = new DatabaseSync(snapshotPath, { readOnly: true })

  try {
    const verification = sqliteVerificationMode(size, integrityCheckMaxBytes)

    if (verification === 'schema-probe') {
      snapshot.prepare('PRAGMA schema_version').get()
      snapshot.prepare('SELECT count(*) FROM sqlite_master').get()

      return verification
    }

    const rows = snapshot.prepare('PRAGMA integrity_check').all() as Array<Record<string, unknown>>

    if (rows.length !== 1 || Object.values(rows[0])[0] !== 'ok') {
      throw new Error('SQLite integrity_check failed for emergency backup')
    }

    return verification
  } finally {
    snapshot.close()
  }
}

function removeFile(filePath: string): void {
  try {
    fs.unlinkSync(filePath)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw error
    }
  }
}
