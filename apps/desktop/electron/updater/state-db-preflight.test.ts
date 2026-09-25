import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { once } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { preflightStateDb, resolveStateDbSnapshotRunner } from './state-db-preflight'

const repository: string = path.resolve(import.meta.dirname, '../../../..')

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
      updateRoot: repository,
      script,
      home,
      isWindows: false,
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
        updateRoot: oldRoot,
        script: path.join(oldRoot, 'hermes_cli', 'backup_sqlite.py'),
        home: oldRoot,
        isWindows: false,
        log: (): void => {}
      })
      stopped = true
    }, /snapshot|pre-flight/)
    assert.equal(stopped, false)
  } finally {
    fs.rmSync(oldRoot, { recursive: true, force: true })
  }
})


test('a PM-managed install with no in-tree venv runs the snapshot via the installation launcher', (): void => {
  // #122991: PM deletes the in-tree venv once a generation is committed, so
  // resolveSourcePython answers null on a managed install and the preflight
  // cancelled every in-app update with "Python not found". The managed rung
  // must resolve the same launcher the update probe (readSourceUpdate) uses.
  const managedRoot: string = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-preflight-'))
  const home: string = path.join(managedRoot, 'home')
  const python: string = process.env.HERMES_PYTHON || 'python3'
  const script: string = path.join(repository, 'hermes_cli', 'backup_sqlite.py')

  try {
    fs.mkdirSync(home, { recursive: true })
    fs.mkdirSync(path.join(managedRoot, 'pm'), { recursive: true })
    fs.mkdirSync(path.join(managedRoot, '.hermes', 'bin'), { recursive: true })

    // A stand-in for the real launcher: translates --run-module the way its
    // embedded bootstrap does (hermes_cli/_launchers.py) and runs the module
    // with the repo on sys.path.
    const launcher: string = path.join(managedRoot, '.hermes', 'bin', 'hermes')
    fs.writeFileSync(
      launcher,
      [
        '#!/bin/sh',
        'if [ "$1" = "--run-module" ]; then',
        '  shift',
        `  exec ${JSON.stringify(python)} -c ${JSON.stringify(
          `import sys, runpy; sys.argv = sys.argv[1:]; sys.path.insert(0, ${JSON.stringify(repository)}); runpy.run_module(sys.argv[0], run_name='__main__', alter_sys=True)`
        )} "$@"`,
        'fi',
        `exec ${JSON.stringify(python)} "$@"`,
        ''
      ].join('\n'),
      { mode: 0o755 }
    )

    const runner = resolveStateDbSnapshotRunner({ python: null, updateRoot: managedRoot, script, home, isWindows: false })

    assert.ok(runner, 'a managed install must resolve the installation launcher when no venv python exists')
    assert.equal(runner.command, launcher)
    assert.deepEqual(runner.args, ['--run-module', 'hermes_cli.backup_sqlite', home])
    assert.equal(runner.viaCmd, false)
    assert.equal(runner.cwd, managedRoot)
    assert.equal(runner.env?.HERMES_HOME, home)
    assert.equal(runner.env?.HERMES_INSTALL_ROOT, managedRoot)

    // The real snapshot through the launcher rung: a busy WAL database must
    // get its emergency .bak through the exact recipe the preflight runs.
    fs.writeFileSync(path.join(home, 'state.db'), '')
    spawnSync(python, ['-c', 'import sqlite3, sys; c = sqlite3.connect(sys.argv[1]); c.execute("CREATE TABLE t (x)"); c.commit(); c.close()', path.join(home, 'state.db')])

    const logs: string[] = []
    preflightStateDb({
      python: null,
      updateRoot: managedRoot,
      script,
      home,
      isWindows: false,
      log: (message: string): void => {
        logs.push(message)
      }
    })

    const backups: string[] = fs.readdirSync(home).filter((name: string): boolean => name.endsWith('.bak'))
    assert.equal(backups.length, 1, logs.join('\n'))
  } finally {
    fs.rmSync(managedRoot, { recursive: true, force: true })
  }
})

test('the launcher rung is only for managed installs; a plain checkout without a venv still refuses', (): void => {
  const plainRoot: string = fs.mkdtempSync(path.join(os.tmpdir(), 'plain-preflight-'))

  try {
    const runner = resolveStateDbSnapshotRunner({
      python: null,
      updateRoot: plainRoot,
      script: path.join(plainRoot, 'hermes_cli', 'backup_sqlite.py'),
      home: plainRoot,
      isWindows: false
    })

    assert.equal(runner, null)
  } finally {
    fs.rmSync(plainRoot, { recursive: true, force: true })
  }
})
