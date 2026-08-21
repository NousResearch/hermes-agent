import fs from 'node:fs'
import { backup, DatabaseSync } from 'node:sqlite'

export async function createVerifiedSqliteBackup(sourcePath: string, destinationPath: string): Promise<void> {
  const source = new DatabaseSync(sourcePath, { readOnly: true })

  try {
    await backup(source, destinationPath)
  } catch (error) {
    removePartialBackup(destinationPath)
    throw error
  } finally {
    source.close()
  }

  let snapshot: DatabaseSync | null = null

  try {
    snapshot = new DatabaseSync(destinationPath, { readOnly: true })
    const rows = snapshot.prepare('PRAGMA integrity_check').all() as Array<Record<string, unknown>>

    if (rows.length !== 1 || Object.values(rows[0])[0] !== 'ok') {
      throw new Error('SQLite integrity_check failed for emergency backup')
    }
  } catch (error) {
    snapshot?.close()
    snapshot = null
    removePartialBackup(destinationPath)
    throw error
  } finally {
    snapshot?.close()
  }
}

function removePartialBackup(destinationPath: string): void {
  try {
    fs.unlinkSync(destinationPath)
  } catch {
    // Nothing to clean up, or another process already removed the partial file.
  }
}
