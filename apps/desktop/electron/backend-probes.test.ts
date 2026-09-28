/**
 * Tests for electron/backend-probes.ts.
 *
 * Run with: node --test electron/backend-probes.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 */

import assert from 'node:assert/strict'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  canImportHermesCli,
  DEFAULT_PROBE_TIMEOUT_MS,
  execProbe,
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

test('execProbe keeps the parent event loop available to the child', async () => {
  let unexpectedSocketError: Error | undefined

  const server = net.createServer(socket => {
    socket.on('error', error => {
      // A successful child exits immediately after reading the sentinel. On
      // Windows that peer close can surface as ECONNRESET on the server side.
      if ((error as NodeJS.ErrnoException).code !== 'ECONNRESET') {
        unexpectedSocketError ??= error
      }
    })
    socket.end('pong')
  })

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })

  const address = server.address()
  assert.ok(address && typeof address === 'object')

  const childScript = `
    const net = require('node:net')
    let reply = ''
    const socket = net.createConnection(${address.port}, '127.0.0.1')
    socket.setEncoding('utf8')
    socket.on('data', (chunk) => { reply += chunk })
    socket.on('end', () => process.exit(reply === 'pong' ? 0 : 1))
    socket.on('error', () => process.exit(1))
  `

  try {
    await execProbe(NODE_BIN, ['-e', childScript], {
      stdio: 'ignore',
      timeout: 5_000,
      windowsHide: true
    })
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close(error => (error ? reject(error) : resolve()))
    })
  }

  assert.ifError(unexpectedSocketError)
})

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

test('verifyHermesCli accepts an actual zero-exit executable', async (): Promise<void> => {
  assert.equal(await verifyHermesCli(NODE_BIN), true)
})

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
