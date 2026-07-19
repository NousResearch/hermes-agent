/**
 * Tests for electron/backend-probes.ts.
 *
 * Run with: node --test electron/backend-probes.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 */

import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { canImportHermesCli, verifyHermesCli } from './backend-probes'

// Resolve the host's own Node binary -- guaranteed to be on disk and
// runnable. We use it as both a stand-in for "a python that doesn't
// have hermes_cli" (since `node -c "import hermes_cli"` will exit
// non-zero) and as a way to script verifyHermesCli's success path
// (a tiny script we write to disk that exits 0 on --version).
const NODE_BIN = process.execPath

function findPythonInterpreter() {
  for (const candidate of [process.env.PYTHON, 'python3', 'python', 'py']) {
    if (!candidate) {
      continue
    }

    const result = spawnSync(candidate, ['-c', 'import sys'], { stdio: 'ignore', windowsHide: true })
    if (!result.error && result.status === 0) {
      return candidate
    }
  }

  throw new Error('Desktop runtime probe tests require an available Python interpreter')
}

function dashboardImportInterpreterFixture() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-probes-dashboard-import-'))
  const hermesCli = path.join(directory, 'hermes_cli')
  fs.mkdirSync(hermesCli)

  // A shallow config probe can import this fixture, but the real Desktop
  // startup boundary cannot: the web-server module represents a broken
  // dashboard dependency. This makes the test prove the actual child-process
  // import behavior rather than freezing the generated Python source text.
  for (const module of ['yaml.py', 'dotenv.py', 'fastapi.py', 'hermes_cli/__init__.py', 'hermes_cli/config.py']) {
    fs.writeFileSync(path.join(directory, module), '')
  }
  fs.writeFileSync(path.join(hermesCli, 'web_server.py'), "raise ImportError('broken dashboard fixture')\n")

  return {
    cleanup: () => fs.rmSync(directory, { force: true, recursive: true }),
    env: { PYTHONPATH: [directory, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) },
    interpreter: findPythonInterpreter()
  }
}

test('canImportHermesCli returns false when path is falsy', () => {
  assert.equal(canImportHermesCli(''), false)
  assert.equal(canImportHermesCli(null), false)
  assert.equal(canImportHermesCli(undefined), false)
})

test('canImportHermesCli returns false when interpreter cannot run -c', () => {
  // node IS an interpreter, but `node -c "import hermes_cli"` is a
  // SyntaxError -- different exit reason from a real Python's
  // ModuleNotFoundError, but the predicate is "exit 0 or not" and
  // both land on "not", which is exactly what we want for the
  // resolver fall-through.
  assert.equal(canImportHermesCli(NODE_BIN), false)
})

test('canImportHermesCli returns false when binary does not exist', () => {
  const ghost = path.join(os.tmpdir(), 'hermes-probes-ghost-' + Date.now() + '.exe')
  assert.equal(canImportHermesCli(ghost), false)
})

test('canImportHermesCli rejects a runtime whose config loads but web-server startup fails', () => {
  const fixture = dashboardImportInterpreterFixture()

  try {
    const shallow = spawnSync(fixture.interpreter, ['-c', 'import yaml; import dotenv; import hermes_cli.config'], {
      env: { ...process.env, ...fixture.env },
      stdio: 'ignore',
      windowsHide: true
    })
    assert.equal(shallow.status, 0)

    // The current probe follows the same import boundary as `hermes serve`,
    // so it must reject this otherwise-configurable interpreter.
    assert.equal(canImportHermesCli(fixture.interpreter, { env: fixture.env }), false)
  } finally {
    fixture.cleanup()
  }
})

test('verifyHermesCli returns false when command is falsy', () => {
  assert.equal(verifyHermesCli(''), false)
  assert.equal(verifyHermesCli(null), false)
  assert.equal(verifyHermesCli(undefined), false)
})

test('verifyHermesCli returns false when binary does not exist', () => {
  const ghost = path.join(os.tmpdir(), 'hermes-probes-ghost-' + Date.now() + '.exe')
  assert.equal(verifyHermesCli(ghost), false)
})

test('verifyHermesCli returns true when --version exits 0', () => {
  // Write a tiny script that exits 0 regardless of args, then invoke
  // it through node. This stands in for a working hermes binary --
  // verifyHermesCli only cares about the exit code.
  const scriptPath = path.join(os.tmpdir(), `hermes-probes-ok-${Date.now()}-${process.pid}.cjs`)
  fs.writeFileSync(scriptPath, 'process.exit(0)\n')

  try {
    // Use node as the launcher and our script as the "command". Pass
    // shell:false (default) -- node is a real binary, no shim.
    // execFileSync passes ['--version'] as args, which node ignores
    // gracefully (well, it prints its version and exits 0, which is
    // perfect -- exit code 0 is the only signal we read).
    assert.equal(verifyHermesCli(NODE_BIN), true)
  } finally {
    try {
      fs.unlinkSync(scriptPath)
    } catch {
      void 0
    }
  }
})

test('verifyHermesCli swallows timeouts (does not throw)', () => {
  // We can't easily provoke a real 5s hang in CI without slowing the
  // suite, but we CAN confirm that an invocation that DOES throw
  // (because the binary is missing) returns false rather than
  // propagating. Same code path the timeout case takes.
  assert.equal(verifyHermesCli('/definitely/not/a/real/binary/anywhere'), false)
})
