import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { Worker } from 'node:worker_threads'

import { afterEach, test } from 'vitest'

import { createVerifiedSqliteBackup, inspectSqliteFootprint, sqliteVerificationMode } from './sqlite-backup'
import { planEmergencyStateDbBackup } from './state-db-update-preflight'

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

    assert.deepEqual(
      sessionRows.map(row => row.id),
      ['wal-only-session']
    )
    assert.deepEqual(
      integrityRows.map(row => row.integrity_check),
      ['ok']
    )
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

  const counter = new Int32Array(new SharedArrayBuffer(4))

  const writer = new Worker(
    `
      const { parentPort, workerData } = require('node:worker_threads')
      const { DatabaseSync } = require('node:sqlite')
      const database = new DatabaseSync(workerData.sourcePath)
      const insert = database.prepare('INSERT INTO writes (value) VALUES (?)')
      let ready = false
      setInterval(() => {
        insert.run(String(Date.now()))
        Atomics.add(new Int32Array(workerData.counter), 0, 1)
        if (!ready) {
          ready = true
          parentPort.postMessage('ready')
        }
      }, 0)
    `,
    { eval: true, workerData: { counter: counter.buffer, sourcePath } }
  )

  try {
    await new Promise<void>((resolve, reject) => {
      writer.once('message', () => resolve())
      writer.once('error', reject)
    })

    const writesBefore = Atomics.load(counter, 0)
    const startedAt = Date.now()
    let completed = false

    try {
      await createVerifiedSqliteBackup(sourcePath, destinationPath, { deadlineMs: 100 })
      completed = true
    } catch (error) {
      assert.match(String(error), /emergency backup exceeded 100ms deadline/)
    }

    assert.ok(Atomics.load(counter, 0) > writesBefore)
    assert.ok(Date.now() - startedAt < 5_000)
    assert.equal(fs.existsSync(`${destinationPath}.partial`), false)

    if (completed) {
      const snapshot = new DatabaseSync(destinationPath, { readOnly: true })

      try {
        const rows = snapshot.prepare('PRAGMA integrity_check').all() as Array<{ integrity_check: string }>

        assert.deepEqual(
          rows.map(row => row.integrity_check),
          ['ok']
        )
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

test('estimates the online snapshot from logical pages and the committed WAL footprint', () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const source = createPopulatedDatabase(sourcePath)

  try {
    source.exec('CREATE TABLE payload (value BLOB)')
    source.prepare('INSERT INTO payload (value) VALUES (?)').run(Buffer.alloc(1024 * 1024))
    const footprint = inspectSqliteFootprint(sourcePath)

    assert.ok(footprint.walBytes > 0)
    assert.ok(footprint.logicalBytes > 0)
    assert.ok(footprint.estimatedBackupBytes >= footprint.logicalBytes)
    assert.ok(footprint.estimatedBackupBytes >= footprint.mainBytes + footprint.walBytes)
    assert.deepEqual(
      planEmergencyStateDbBackup(
        { backupKeep: 5, mode: 'quick', quickKeep: 1, quickMaxFileSize: footprint.mainBytes },
        footprint.estimatedBackupBytes
      ),
      { reason: 'quick-size-cap', status: 'skipped' }
    )
  } finally {
    source.close()
  }
})

test('keeps the final name absent until verification has completed', async () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const destinationPath = path.join(dir, 'state.db.bak')
  const source = createPopulatedDatabase(sourcePath)
  const phases: string[] = []

  try {
    await createVerifiedSqliteBackup(sourcePath, destinationPath, {
      onPhase: phase => {
        phases.push(phase)

        if (phase !== 'published') {
          assert.equal(fs.existsSync(destinationPath), false)
          assert.equal(fs.existsSync(`${destinationPath}.partial`), true)
        }
      }
    })

    assert.deepEqual(phases, ['backup-complete', 'verified', 'published'])
    assert.equal(fs.existsSync(destinationPath), true)
  } finally {
    source.close()
  }
})

test('uses the bounded structural worker probe above the configured deep-check boundary', async () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const destinationPath = path.join(dir, 'state.db.bak')
  const source = createPopulatedDatabase(sourcePath)

  try {
    const result = await createVerifiedSqliteBackup(sourcePath, destinationPath, {
      integrityCheckMaxBytes: 1,
      verificationDeadlineMs: 5_000
    })

    assert.equal(result.verification, 'schema-probe')
    assert.equal(fs.existsSync(destinationPath), true)
    assert.equal(fs.existsSync(`${destinationPath}.partial`), false)
  } finally {
    source.close()
  }
})

test('preserves a pre-existing final byte-for-byte when publication cannot claim the name', async () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const destinationPath = path.join(dir, 'state.db.bak')
  const source = createPopulatedDatabase(sourcePath)
  const sentinel = Buffer.from('existing recovery point')
  fs.writeFileSync(destinationPath, sentinel)

  try {
    await assert.rejects(createVerifiedSqliteBackup(sourcePath, destinationPath), /already exists/)
    assert.deepEqual(fs.readFileSync(destinationPath), sentinel)
    assert.equal(fs.existsSync(`${destinationPath}.partial`), false)
  } finally {
    source.close()
  }
})

test('preserves a final that appears after verification instead of overwriting it', async () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const destinationPath = path.join(dir, 'state.db.bak')
  const source = createPopulatedDatabase(sourcePath)
  const sentinel = Buffer.from('racing recovery point')

  try {
    await assert.rejects(
      createVerifiedSqliteBackup(sourcePath, destinationPath, {
        onPhase: phase => {
          if (phase === 'verified') {
            fs.writeFileSync(destinationPath, sentinel)
          }
        }
      }),
      /already exists/
    )
    assert.deepEqual(fs.readFileSync(destinationPath), sentinel)
    assert.equal(fs.existsSync(`${destinationPath}.partial`), false)
  } finally {
    source.close()
  }
})

test('a forced worker termination leaves no final and the next attempt replaces its stale partial', async () => {
  const dir = tempDir()
  const sourcePath = path.join(dir, 'state.db')
  const destinationPath = path.join(dir, 'state.db.bak')
  const source = new DatabaseSync(sourcePath)
  source.exec('CREATE TABLE payload (value BLOB)')
  const insert = source.prepare('INSERT INTO payload (value) VALUES (?)')

  for (let index = 0; index < 16; index += 1) {
    insert.run(Buffer.alloc(1024 * 1024, index))
  }

  source.close()

  const worker = new Worker(
    `
      const { parentPort, workerData } = require('node:worker_threads')
      const { backup, DatabaseSync } = require('node:sqlite')
      const source = new DatabaseSync(workerData.source)
      const wait = new Int32Array(new SharedArrayBuffer(4))
      backup(source, workerData.partial, {
        rate: 1,
        progress() {
          parentPort.postMessage('partial-created')
          Atomics.wait(wait, 0, 0, 10000)
        }
      }).catch(error => parentPort.postMessage(String(error)))
    `,
    { eval: true, workerData: { partial: `${destinationPath}.partial`, source: sourcePath } }
  )

  await new Promise<void>((resolve, reject) => {
    worker.once('message', () => resolve())
    worker.once('error', reject)
  })
  await worker.terminate()

  assert.equal(fs.existsSync(destinationPath), false)
  assert.equal(fs.existsSync(`${destinationPath}.partial`), true)

  const result = await createVerifiedSqliteBackup(sourcePath, destinationPath)
  assert.ok(result.size > 0)
  assert.equal(fs.existsSync(destinationPath), true)
  assert.equal(fs.existsSync(`${destinationPath}.partial`), false)
})
