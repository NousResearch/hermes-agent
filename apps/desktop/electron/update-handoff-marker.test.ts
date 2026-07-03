/**
 * Tests for electron/update-handoff-marker.ts — the durable handshake marker
 * for the Tauri updater hand-off (#57645).
 *
 * When the desktop app clicks "Update now" and a staged `hermes-setup` binary
 * exists, it spawns that updater detached and quits. If the updater fails
 * silently, the relaunched instance must detect that the git SHA did NOT
 * change and surface a closeable error instead of silently reverting.
 *
 * Run with: node --test electron/update-handoff-marker.test.ts
 */

import fs from 'fs'
import assert from 'node:assert/strict'
import test from 'node:test'
import os from 'os'
import path from 'path'

import {
  handoffMarkerPath,
  writeHermesUpdateHandoff,
  readUpdateHandoffResult,
  HANDOFF_MARKER_MAX_AGE_MS
} from './update-handoff-marker'

function tmpHome(tag) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-handoff-${tag}-`))
  return dir
}

function writeMarker(home, sha, version, at) {
  const markerPath = handoffMarkerPath(home)
  fs.mkdirSync(path.dirname(markerPath), { recursive: true })
  fs.writeFileSync(markerPath, JSON.stringify({ sha, version, at }))
}

// Injectable SHA resolvers
const shaOf = s => () => s
const noSha = () => ''

test('absent marker => no pending hand-off', () => {
  const home = tmpHome('absent')
  const result = readUpdateHandoffResult(home, '/fake/root', { resolveSha: noSha })
  assert.equal(result.pending, false)
  assert.equal(result.failed, undefined)
})

test('SHA changed after hand-off => update succeeded, not pending', () => {
  const home = tmpHome('succeeded')
  const now = Date.now()
  writeMarker(home, 'aaa111', '0.18.0', now)
  const result = readUpdateHandoffResult(home, '/fake/root', {
    resolveSha: shaOf('bbb222'),
    now: () => now + 5000
  })
  assert.equal(result.pending, false)
  assert.equal(result.succeeded, true)
  assert.equal(result.failed, undefined)
  // Marker is pruned after being read (one-shot).
  assert.ok(!fs.existsSync(handoffMarkerPath(home)))
})

test('SHA unchanged after hand-off => update failed, pending', () => {
  const home = tmpHome('failed')
  const now = Date.now()
  writeMarker(home, 'aaa111', '0.18.0', now)
  const result = readUpdateHandoffResult(home, '/fake/root', {
    resolveSha: shaOf('aaa111'),
    now: () => now + 5000
  })
  assert.equal(result.pending, true)
  assert.equal(result.failed, true)
  assert.equal(result.recordedSha, 'aaa111')
  assert.equal(result.currentSha, 'aaa111')
  assert.equal(result.recordedVersion, '0.18.0')
  // Marker is pruned even on failure (one-shot, not a lingering state).
  assert.ok(!fs.existsSync(handoffMarkerPath(home)))
})

test('stale marker (past age ceiling) => no pending hand-off, pruned', () => {
  const home = tmpHome('stale')
  const now = 1_000_000_000_000
  writeMarker(home, 'aaa111', '0.18.0', now - HANDOFF_MARKER_MAX_AGE_MS - 60_000)
  const result = readUpdateHandoffResult(home, '/fake/root', {
    resolveSha: shaOf('aaa111'),
    now: () => now
  })
  assert.equal(result.pending, false)
  assert.equal(result.failed, undefined)
  assert.ok(!fs.existsSync(handoffMarkerPath(home)))
})

test('malformed marker => no pending hand-off, pruned', () => {
  const home = tmpHome('malformed')
  fs.mkdirSync(home, { recursive: true })
  fs.writeFileSync(handoffMarkerPath(home), 'not-json{')
  const result = readUpdateHandoffResult(home, '/fake/root', { resolveSha: noSha })
  assert.equal(result.pending, false)
  assert.ok(!fs.existsSync(handoffMarkerPath(home)))
})

test('writeHermesUpdateHandoff writes a valid JSON marker', () => {
  const home = tmpHome('write')
  writeHermesUpdateHandoff(home, { sha: 'abc123', version: '0.18.0' })
  const raw = fs.readFileSync(handoffMarkerPath(home), 'utf8')
  const marker = JSON.parse(raw)
  assert.equal(marker.sha, 'abc123')
  assert.equal(marker.version, '0.18.0')
  assert.ok(Number.isFinite(marker.at))
})

test('writeHermesUpdateHandoff with empty SHA still works (version-only check)', () => {
  const home = tmpHome('write-empty-sha')
  writeHermesUpdateHandoff(home, { sha: '', version: '0.18.0' })
  // With empty SHA on both sides, the update is treated as failed (SHA match
  // on empty strings) — this is the conservative default: if we can't resolve
  // the SHA, we can't prove the update took.
  const result = readUpdateHandoffResult(home, '/fake/root', {
    resolveSha: noSha
  })
  assert.equal(result.pending, true)
  assert.equal(result.failed, true)
})

test('writeHermesUpdateHandoff + readUpdateHandoffResult round-trip', () => {
  const home = tmpHome('roundtrip')
  writeHermesUpdateHandoff(home, { sha: 'sha-before', version: '0.17.0' })
  // Simulate update succeeded: SHA changed.
  const result = readUpdateHandoffResult(home, '/fake/root', {
    resolveSha: shaOf('sha-after')
  })
  assert.equal(result.pending, false)
  assert.equal(result.succeeded, true)
  assert.ok(!fs.existsSync(handoffMarkerPath(home)))
})
