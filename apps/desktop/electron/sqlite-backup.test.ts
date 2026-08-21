import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'

import { afterEach, test } from 'vitest'

import { createVerifiedSqliteBackup } from './sqlite-backup'

const tempDirs: string[] = []

function tempDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sqlite-backup-'))

  tempDirs.push(dir)

  return dir
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('includes committed WAL transactions in the verified snapshot', async () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const destinationPath = path.join(dir, 'state.db.bak')
  const source = new DatabaseSync(sourcePath)

  source.exec('PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0; CREATE TABLE sessions (id TEXT PRIMARY KEY);')
  source.prepare('INSERT INTO sessions (id) VALUES (?)').run('wal-only-session')

  assert.ok(fs.existsSync(`${sourcePath}-wal`))
  await createVerifiedSqliteBackup(sourcePath, destinationPath)

  const snapshot = new DatabaseSync(destinationPath, { readOnly: true })

  try {
    const sessionRows = snapshot.prepare('SELECT id FROM sessions').all() as Array<{ id: string }>
    const integrityRows = snapshot.prepare('PRAGMA integrity_check').all() as Array<{ integrity_check: string }>

    assert.deepEqual(sessionRows.map(row => row.id), ['wal-only-session'])
    assert.deepEqual(integrityRows.map(row => row.integrity_check), ['ok'])
  } finally {
    snapshot.close()
    source.close()
  }
})
