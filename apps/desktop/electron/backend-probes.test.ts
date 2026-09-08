/**
 * Tests for electron/backend-probes.ts.
 *
 * Run with: node --test electron/backend-probes.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import { createServer } from 'node:net'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  canImportHermesCli,
  DEFAULT_PROBE_TIMEOUT_MS,
  execProbe,
  hermesRuntimeImportProbe,
  PROBE_TIMEOUT_MS,
  resolveProbeTimeoutMs,
  shouldTrustHermesOverride,
  verifyHermesCli
} from './backend-probes'

// Resolve the host's own Node binary -- guaranteed to be on disk and
// runnable. We use it as both a stand-in for "a python that doesn't
// have hermes_cli" (since `node -c "import hermes_cli"` will exit
// non-zero) and as a way to script verifyHermesCli's success path
// (a tiny script we write to disk that exits 0 on --version).
const NODE_BIN = process.execPath

test('runtime probe lets the parent service IO while its child is running', async () => {
  let accepted = 0

  const server = createServer(socket => {
    accepted += 1
    socket.on('error', () => {})
    socket.end('ready')
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')

  const script = `
    const socket = require('node:net').connect(${address.port}, '127.0.0.1');
    socket.on('data', () => socket.destroy());
    socket.on('error', () => process.exit(1));
  `

  try {
    // A synchronous probe prevents the parent from accepting this connection,
    // so both timeout attempts fail even though the child is healthy (#103786).
    await execProbe(NODE_BIN, ['-e', script], { stdio: 'ignore', timeout: 2000 })
    assert.equal(accepted, 1)
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
}, 10_000)

test('canImportHermesCli returns false when path is falsy', async () => {
  assert.equal(await canImportHermesCli(''), false)
  assert.equal(await canImportHermesCli(null), false)
  assert.equal(await canImportHermesCli(undefined), false)
})

test('canImportHermesCli returns false when interpreter cannot run -c', async () => {
  // node IS an interpreter, but `node -c "import hermes_cli"` is a
  // SyntaxError -- different exit reason from a real Python's
  // ModuleNotFoundError, but the predicate is "exit 0 or not" and
  // both land on "not", which is exactly what we want for the
  // resolver fall-through.
  assert.equal(await canImportHermesCli(NODE_BIN), false)
})

test('canImportHermesCli returns false when binary does not exist', async () => {
  const ghost = path.join(os.tmpdir(), 'hermes-probes-ghost-' + Date.now() + '.exe')
  assert.equal(await canImportHermesCli(ghost), false)
})

test('hermes runtime import probe checks config dependencies', () => {
  const probe = hermesRuntimeImportProbe()
  assert.match(probe, /\bimport yaml\b/)
  // dotenv is the first third-party import on the CLI boot path
  // (hermes_cli/env_loader.py); a mid-update venv missing python-dotenv
  // passed the old probe and produced an unrecoverable boot loop.
  assert.match(probe, /\bimport dotenv\b/)
  assert.match(probe, /\bimport hermes_cli\.config\b/)
})

test('explicit Hermes override is authoritative', () => {
  assert.equal(shouldTrustHermesOverride('/nix/store/abc/bin/hermes'), true)
})

test('empty Hermes override is not authoritative', () => {
  assert.equal(shouldTrustHermesOverride(''), false)
  assert.equal(shouldTrustHermesOverride(undefined), false)
})

test('verifyHermesCli returns false when command is falsy', async () => {
  assert.equal(await verifyHermesCli(''), false)
  assert.equal(await verifyHermesCli(null), false)
  assert.equal(await verifyHermesCli(undefined), false)
})

test('verifyHermesCli returns false when binary does not exist', async () => {
  const ghost = path.join(os.tmpdir(), 'hermes-probes-ghost-' + Date.now() + '.exe')
  assert.equal(await verifyHermesCli(ghost), false)
})

test('verifyHermesCli returns true when --version exits 0', async () => {
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
    assert.equal(await verifyHermesCli(NODE_BIN), true)
  } finally {
    try {
      fs.unlinkSync(scriptPath)
    } catch {
      void 0
    }
  }
})

test('probe retries only timeouts, including a child that exits zero when terminated', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-probe-retry-'))

  try {
    for (const mode of ['recover', 'timeout', 'nonzero', 'clean-timeout']) {
      const record = path.join(root, mode)

      const script = `
        const fs = require('node:fs');
        fs.appendFileSync(${JSON.stringify(record)}, 'attempt\\n');
        const count = fs.readFileSync(${JSON.stringify(record)}, 'utf8').trim().split('\\n').length;
        if (${JSON.stringify(mode)} === 'nonzero') process.exit(1);
        if (${JSON.stringify(mode)} === 'recover' && count === 2) process.exit(0);
        process.on('SIGTERM', () => process.exit(${mode === 'clean-timeout' ? 0 : 1}));
        setInterval(() => {}, 1000);
      `

      const result = execProbe(NODE_BIN, ['-e', script], { stdio: 'ignore', timeout: 2000 })

      if (mode === 'recover') {
        await result
      } else {
        await assert.rejects(result, undefined, mode)
      }

      assert.equal(fs.readFileSync(record, 'utf8').trim().split('\n').length, mode === 'nonzero' ? 1 : 2, mode)
    }
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}, 20_000)

test('default probe timeout is 15s (not the old 5s death-loop value)', () => {
  assert.equal(DEFAULT_PROBE_TIMEOUT_MS, 15_000)
  // Module constant uses process.env at load time; with no override it
  // matches the default (tests run without HERMES_PROBE_TIMEOUT_MS).
  assert.equal(PROBE_TIMEOUT_MS, DEFAULT_PROBE_TIMEOUT_MS)
})

test('resolveProbeTimeoutMs honours HERMES_PROBE_TIMEOUT_MS', () => {
  assert.equal(resolveProbeTimeoutMs({}), DEFAULT_PROBE_TIMEOUT_MS)
  assert.equal(resolveProbeTimeoutMs({ HERMES_PROBE_TIMEOUT_MS: '30000' }), 30_000)
  assert.equal(resolveProbeTimeoutMs({ HERMES_PROBE_TIMEOUT_MS: '0' }), DEFAULT_PROBE_TIMEOUT_MS)
  assert.equal(resolveProbeTimeoutMs({ HERMES_PROBE_TIMEOUT_MS: 'nope' }), DEFAULT_PROBE_TIMEOUT_MS)
  // Cap runaway values
  assert.equal(resolveProbeTimeoutMs({ HERMES_PROBE_TIMEOUT_MS: '999999' }), 120_000)
})
