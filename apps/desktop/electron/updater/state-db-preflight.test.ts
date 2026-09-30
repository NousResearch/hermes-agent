import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { once } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { preflightStateDb, stateDbPreflightTimeoutMs } from './state-db-preflight'

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
    await preflightStateDb({
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

test('a managed installation runs the snapshot through the installation launcher', async (): Promise<void> => {
  const home: string = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-preflight-'))
  const shims: string = fs.mkdtempSync(path.join(os.tmpdir(), 'launcher-shim-'))
  const python: string = process.env.HERMES_PYTHON || 'python3'
  const script: string = fileURLToPath(new URL('../../../../hermes_cli/backup_sqlite.py', import.meta.url))

  // Stand-in for the installation launcher under `.hermes/bin`: it must accept
  // exactly what the runtime passes it — `--run-module hermes_cli.backup_sqlite
  // <home>` — and publish the snapshot like the real launcher does.
  const shim: string = path.join(shims, process.platform === 'win32' ? 'hermes.cmd' : 'hermes')
  fs.writeFileSync(
    shim,
    process.platform === 'win32'
      ? `@echo off\r\n"${python}" -I -S "${script}" %3\r\n`
      : `#!/bin/sh\nexec "${python}" -I -S "${script}" "$3"\n`
  )

  if (process.platform !== 'win32') {
    fs.chmodSync(shim, 0o755)
  }

  const logs: string[] = []

  try {
    const created = spawnSync(
      python,
      [
        '-I',
        '-S',
        '-c',
        "import sqlite3, sys; c = sqlite3.connect(sys.argv[1]); c.execute('CREATE TABLE t (x)'); c.commit(); c.close()",
        path.join(home, 'state.db')
      ],
      { encoding: 'utf8' }
    )

    assert.equal(created.status, 0, created.stderr)

    await preflightStateDb({
      python: null,
      launcher: shim,
      script,
      home,
      log: (message: string): void => {
        logs.push(message)
      }
    })

    const backups: string[] = fs.readdirSync(home).filter((name: string): boolean => name.endsWith('.bak'))
    assert.equal(backups.length, 1, logs.join('\n'))
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
    fs.rmSync(shims, { recursive: true, force: true })
  }
})

test('an older selected checkout without the snapshot helper refuses before backend stop', async (): Promise<void> => {
  const oldRoot: string = fs.mkdtempSync(path.join(os.tmpdir(), 'old-preflight-'))
  let stopped = false

  try {
    await assert.rejects(async (): Promise<void> => {
      await preflightStateDb({
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

test('the deadline grows with the main database and WAL, but stays bounded', (): void => {
  const home: string = fs.mkdtempSync(path.join(os.tmpdir(), 'preflight-budget-'))

  try {
    const floor = stateDbPreflightTimeoutMs(home)

    for (const name of ['state.db', 'state.db-wal']) {
      fs.writeFileSync(path.join(home, name), '')
      fs.truncateSync(path.join(home, name), 2 * 1024 ** 3)
    }

    const withWal = stateDbPreflightTimeoutMs(home)
    fs.unlinkSync(path.join(home, 'state.db-wal'))
    const withoutWal = stateDbPreflightTimeoutMs(home)
    assert.ok(floor >= 5 * 60_000)
    assert.ok(withWal > withoutWal && withoutWal > floor)
    fs.truncateSync(path.join(home, 'state.db'), 100 * 1024 ** 3)
    assert.ok(stateDbPreflightTimeoutMs(home) <= 30 * 60_000)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('the event loop stays available while a real snapshot process is running', async (): Promise<void> => {
  const home: string = fs.mkdtempSync(path.join(os.tmpdir(), 'preflight-responsive-'))
  const script = path.join(home, 'snapshot.py')
  fs.writeFileSync(script, `import pathlib, sys, time
home = pathlib.Path(sys.argv[1])
deadline = time.monotonic() + 3
while not (home / 'release').exists():
    if time.monotonic() > deadline:
        raise RuntimeError('parent event loop was blocked')
    time.sleep(0.01)
print('snapshot finished')
`)
  const timer = setTimeout(() => fs.writeFileSync(path.join(home, 'release'), ''), 100)

  try {
    await preflightStateDb({ python: process.env.HERMES_PYTHON || 'python3', script, home, log: (): void => {} })
  } finally {
    clearTimeout(timer)
    fs.rmSync(home, { recursive: true, force: true })
  }
})
