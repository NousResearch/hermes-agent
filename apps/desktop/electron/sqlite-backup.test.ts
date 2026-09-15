import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { Worker } from 'node:worker_threads'

import { afterEach, test } from 'vitest'

import { createVerifiedSqliteBackup, sqliteVerificationMode } from './sqlite-backup'

const tempDirs: string[] = []

function tempDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sqlite-backup-'))

  tempDirs.push(dir)

  return dir
}

function createPopulatedDatabase(sourcePath: string): DatabaseSync {
  const source = new DatabaseSync(sourcePath)

  source.exec('PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0; CREATE TABLE sessions (id TEXT PRIMARY KEY);')
  source.prepare('INSERT INTO sessions (id) VALUES (?)').run('wal-only-session')

  return source
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('uses the default 2 GiB boundary for deep verification', () => {
  assert.equal(sqliteVerificationMode(2 * 1024 * 1024 * 1024), 'integrity-check')
  assert.equal(sqliteVerificationMode(2 * 1024 * 1024 * 1024 + 1), 'schema-probe')
})

test('includes committed WAL transactions and publishes only the verified snapshot', async () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const destinationPath = path.join(dir, 'state.db.bak')
  const source = createPopulatedDatabase(sourcePath)

  assert.ok(fs.existsSync(`${sourcePath}-wal`))
  const result = await createVerifiedSqliteBackup(sourcePath, destinationPath)

  assert.equal(result.verification, 'integrity-check')
  assert.ok(result.size > 0)
  assert.equal(fs.existsSync(`${destinationPath}.partial`), false)

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

test('uses the bounded structural probe above the deep-check size limit', async () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const destinationPath = path.join(dir, 'state.db.bak')
  const source = createPopulatedDatabase(sourcePath)

  try {
    const result = await createVerifiedSqliteBackup(sourcePath, destinationPath, { integrityCheckMaxBytes: 1 })

    assert.equal(result.verification, 'schema-probe')
    assert.ok(fs.existsSync(destinationPath))
  } finally {
    source.close()
  }
})

test('aborts within the backup deadline without publishing a partial snapshot', async () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const destinationPath = path.join(dir, 'state.db.bak')
  const source = createPopulatedDatabase(sourcePath)

  try {
    source.exec('CREATE TABLE payload (value BLOB)')
    source.prepare('INSERT INTO payload (value) VALUES (?)').run(Buffer.alloc(2 * 1024 * 1024))
    fs.writeFileSync(`${destinationPath}.partial`, 'stale partial')

    await assert.rejects(
      createVerifiedSqliteBackup(sourcePath, destinationPath, { deadlineMs: 0 }),
      /emergency backup exceeded 0ms deadline/
    )
    assert.equal(fs.existsSync(destinationPath), false)
    assert.equal(fs.existsSync(`${destinationPath}.partial`), false)
  } finally {
    source.close()
  }
})

test('reaches a bounded outcome while another connection continuously writes', async () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const destinationPath = path.join(dir, 'state.db.bak')
  const source = createPopulatedDatabase(sourcePath)

  source.exec('CREATE TABLE writes (value TEXT); CREATE TABLE payload (value BLOB)')
  source.prepare('INSERT INTO payload (value) VALUES (?)').run(Buffer.alloc(4 * 1024 * 1024))
  source.close()

  const writer = new Worker(
    `
      const { parentPort, workerData } = require('node:worker_threads')
      const { DatabaseSync } = require('node:sqlite')
      const database = new DatabaseSync(workerData)
      const insert = database.prepare('INSERT INTO writes (value) VALUES (?)')
      parentPort.postMessage('ready')
      setInterval(() => insert.run(String(Date.now())), 0)
    `,
    { eval: true, workerData: sourcePath }
  )

  try {
    await new Promise<void>((resolve, reject) => {
      writer.once('message', () => resolve())
      writer.once('error', reject)
    })

    const startedAt = Date.now()
    let completed = false

    try {
      await createVerifiedSqliteBackup(sourcePath, destinationPath, { deadlineMs: 100 })
      completed = true
    } catch {
      // A timeout is also a valid bounded outcome under sustained writes.
    }

    assert.ok(Date.now() - startedAt < 5_000)
    assert.equal(fs.existsSync(`${destinationPath}.partial`), false)

    if (completed) {
      const snapshot = new DatabaseSync(destinationPath, { readOnly: true })

      try {
        const rows = snapshot.prepare('PRAGMA integrity_check').all() as Array<{ integrity_check: string }>

        assert.deepEqual(rows.map(row => row.integrity_check), ['ok'])
      } finally {
        snapshot.close()
      }
    } else {
      assert.equal(fs.existsSync(destinationPath), false)
    }
  } finally {
    await writer.terminate()
  }
})

test('removes a partial when the backup operation rejects', async () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const destinationPath = path.join(dir, 'missing', 'state.db.bak')
  const partialPath = `${destinationPath}.partial`
  const source = createPopulatedDatabase(sourcePath)

  try {
    await assert.rejects(createVerifiedSqliteBackup(sourcePath, destinationPath))
    assert.equal(fs.existsSync(destinationPath), false)
    assert.equal(fs.existsSync(partialPath), false)
  } finally {
    source.close()
  }
})

test('rejects a corrupt snapshot and removes both publication paths', async () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const destinationPath = path.join(dir, 'state.db.bak')
  const source = new DatabaseSync(sourcePath)

  source.exec('CREATE TABLE payload (value BLOB)')
  const insert = source.prepare('INSERT INTO payload (value) VALUES (?)')

  for (let index = 0; index < 50; index += 1) {
    insert.run(Buffer.alloc(4096, index))
  }

  source.close()

  const fd = fs.openSync(sourcePath, 'r+')

  try {
    fs.writeSync(fd, Buffer.alloc(4096, 0xff), 0, 4096, 2 * 4096)
  } finally {
    fs.closeSync(fd)
  }

  await assert.rejects(createVerifiedSqliteBackup(sourcePath, destinationPath), /integrity_check|malformed/)
  assert.equal(fs.existsSync(destinationPath), false)
  assert.equal(fs.existsSync(`${destinationPath}.partial`), false)
})
