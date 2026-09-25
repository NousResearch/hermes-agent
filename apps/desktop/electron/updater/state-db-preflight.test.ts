import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { once } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { preflightStateDb } from './state-db-preflight'

test('the desktop preflight publishes committed WAL rows before its caller can stop the backend', async (): Promise<void> => {
  const home: string = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-db-'))
  const python: string = process.env.HERMES_PYTHON || 'python3'
  const script: string = fileURLToPath(new URL('../../../../hermes_cli/backup_sqlite.py', import.meta.url))

  const child = spawn(
    python,
    [
      '-I',
      '-S',
      '-u',
      '-c',
      `
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.execute('PRAGMA journal_mode=WAL')
c.execute('PRAGMA wal_autocheckpoint=0')
c.execute('CREATE TABLE messages (body TEXT)')
c.commit()
c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
c.execute("INSERT INTO messages VALUES ('pending in WAL')")
c.commit()
print('ready', flush=True)
sys.stdin.readline()
c.close()
`,
      path.join(home, 'state.db')
    ],
    { stdio: ['pipe', 'pipe', 'pipe'] }
  )

  const logs: string[] = []

  try {
    await once(child.stdout!, 'data')
    preflightStateDb({
      python,
      script,
      home,
      log: (message: string): void => {
        logs.push(message)
      }
    })
    assert.equal(child.exitCode, null)
    const backups: string[] = fs.readdirSync(home).filter((name: string): boolean => name.endsWith('.bak'))
    assert.equal(backups.length, 1, logs.join('\n'))

    const verify = spawnSync(
      python,
      [
        '-I',
        '-S',
        '-c',
        `
import sqlite3, sys
with sqlite3.connect(sys.argv[1]) as c:
    assert c.execute('SELECT body FROM messages').fetchall() == [('pending in WAL',)]
`,
        path.join(home, backups[0]!)
      ],
      { encoding: 'utf8' }
    )

    assert.equal(verify.status, 0, verify.stderr)
    const exited = once(child, 'exit')
    child.stdin!.end('\n')
    await exited
  } finally {
    if (child.exitCode === null && child.signalCode === null) {
      child.kill('SIGKILL')
      await once(child, 'exit')
    }

    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('an older selected checkout without the snapshot helper refuses before backend stop', (): void => {
  const oldRoot: string = fs.mkdtempSync(path.join(os.tmpdir(), 'old-preflight-'))
  let stopped = false

  try {
    assert.throws((): void => {
      preflightStateDb({
        python: process.env.HERMES_PYTHON || 'python3',
        script: path.join(oldRoot, 'hermes_cli', 'backup_sqlite.py'),
        home: oldRoot,
        log: (): void => {}
      })
      stopped = true
    }, /snapshot|pre-flight/)
    assert.equal(stopped, false)
  } finally {
    fs.rmSync(oldRoot, { recursive: true, force: true })
  }
})
