/**
 * Tests for electron/backend-probes.ts.
 *
 * Run with: node --test electron/backend-probes.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 */

import assert from 'node:assert/strict'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  canImportHermesCli,
  DEFAULT_PROBE_TIMEOUT_MS,
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

test('explicit Hermes override is authoritative', () => {
  assert.equal(shouldTrustHermesOverride('/nix/store/abc/bin/hermes'), true)
})

test('empty Hermes override is not authoritative', () => {
  assert.equal(shouldTrustHermesOverride(''), false)
  assert.equal(shouldTrustHermesOverride(undefined), false)
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

test('verifyHermesCli accepts an actual zero-exit executable', (): void => {
  assert.equal(verifyHermesCli(NODE_BIN), true)
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
