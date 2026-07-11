/**
 * Tests for electron/render-cache.ts.
 *
 * Run with: node --test electron/render-cache.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Covers the Phase-1 invariants from the startup-latency spec:
 * debounced atomic writes, 0600 file mode (AC6a), D5 envelope discard rules,
 * I3 corrupt-cache fail-open, I4b eviction (cull-on-delete, LRU count cap,
 * boot sweep against live sessions), and the negative write-failure path.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import {
  MAX_TRANSCRIPT_FILES,
  RENDER_CACHE_SCHEMA,
  RenderCache,
  sessionIdFromTranscriptFile
} from './render-cache.ts'

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'render-cache-test-'))
}

function makeCache(dir: string, overrides: any = {}) {
  return new RenderCache({
    dir,
    appVersion: '0.17.0',
    gatewayUrl: 'http://studio:9119',
    debounceMs: 0, // flush immediately in tests unless a test overrides
    ...overrides
  })
}

// ------------------------------------------------------------------ write path

test('putSessions + flush writes an enveloped file, atomically, mode 0600', () => {
  const dir = tmpDir()
  const cache = makeCache(dir)
  cache.putSessions({ total: 2, sessions: [{ id: 'a' }, { id: 'b' }] })
  cache.flush()

  const target = path.join(dir, 'sessions.json')
  assert.ok(fs.existsSync(target), 'sessions.json exists')
  // no tmp files left behind (atomic rename completed)
  assert.deepEqual(
    fs.readdirSync(dir).filter(n => n.includes('.tmp-')),
    []
  )
  // AC6a: mode is 0600 — no broader than state.db's SQLite default.
  const mode = fs.statSync(target).mode & 0o777
  assert.equal(mode, 0o600)

  const parsed = JSON.parse(fs.readFileSync(target, 'utf8'))
  assert.equal(parsed.schema, RENDER_CACHE_SCHEMA)
  assert.equal(parsed.gatewayUrl, 'http://studio:9119')
  assert.equal(parsed.data.total, 2)
})

test('debounce: writes are deferred until the window elapses, then batched', async () => {
  const dir = tmpDir()
  const cache = makeCache(dir, { debounceMs: 40 })
  cache.putStatus({ ok: 1 })
  cache.putSessions({ total: 0, sessions: [] })
  // nothing on disk yet — the debounce window is open
  assert.equal(fs.existsSync(path.join(dir, 'status.json')), false)
  await new Promise(r => setTimeout(r, 90))
  assert.ok(fs.existsSync(path.join(dir, 'status.json')), 'status flushed after debounce')
  assert.ok(fs.existsSync(path.join(dir, 'sessions.json')), 'sessions flushed in same batch')
})

test('flush() is synchronous and drains pending immediately (before-quit path)', () => {
  const dir = tmpDir()
  const cache = makeCache(dir, { debounceMs: 60_000 }) // window far in the future
  cache.putStatus({ ok: 1 })
  assert.equal(fs.existsSync(path.join(dir, 'status.json')), false)
  cache.flush() // the before-quit call
  assert.ok(fs.existsSync(path.join(dir, 'status.json')), 'flush-on-quit persisted the tail')
})

test('putTranscript caps rows at the row cap and refuses path-ish session ids', () => {
  const dir = tmpDir()
  const cache = makeCache(dir)
  const rows = Array.from({ length: 500 }, (_, i) => ({ i }))
  cache.putTranscript('sess_ok-1', rows)
  cache.putTranscript('../evil', rows) // must be refused
  cache.putTranscript('', rows) // must be refused
  cache.flush()
  const files = fs.readdirSync(dir)
  assert.deepEqual(files, ['transcript-sess_ok-1.json'])
  const parsed = JSON.parse(fs.readFileSync(path.join(dir, 'transcript-sess_ok-1.json'), 'utf8'))
  assert.equal(parsed.data.rows.length, 200)
  // the cap keeps the TAIL (latest rows), matching what the renderer paints
  assert.equal(parsed.data.rows[199].i, 499)
})

test('write failure (unwritable dir) is logged, never throws (I3 negative)', () => {
  const logs: string[] = []
  // A FILE where the dir should be → mkdir/write fails deterministically.
  const blocker = path.join(tmpDir(), 'not-a-dir')
  fs.writeFileSync(blocker, 'x')
  const cache = makeCache(blocker, { log: (l: string) => logs.push(l) })
  cache.putStatus({ ok: 1 })
  assert.doesNotThrow(() => cache.flush())
  assert.ok(
    logs.some(l => l.includes('write failed')),
    `expected a write-failed log, got: ${JSON.stringify(logs)}`
  )
})

// ------------------------------------------------------------------- read path

test('read round-trip returns the inner data', () => {
  const dir = tmpDir()
  const cache = makeCache(dir)
  cache.putSessions({ total: 1, sessions: [{ id: 's1' }] })
  cache.putTranscript('s1', [{ text: 'hello' }])
  cache.flush()
  assert.deepEqual(cache.readSessions<any>()!.sessions, [{ id: 's1' }])
  assert.deepEqual(cache.readTranscript<any>('s1')!.rows, [{ text: 'hello' }])
})

test('missing file reads as null (I3 fail-open)', () => {
  const cache = makeCache(tmpDir())
  assert.equal(cache.readSessions(), null)
  assert.equal(cache.readTranscript('nope'), null)
})

test('corrupt file reads as null, never throws (I3)', () => {
  const dir = tmpDir()
  const cache = makeCache(dir)
  fs.writeFileSync(path.join(dir, 'sessions.json'), '{ definitely not json')
  assert.equal(cache.readSessions(), null)
})

test('schema mismatch discards (D5)', () => {
  const dir = tmpDir()
  const cache = makeCache(dir)
  fs.writeFileSync(
    path.join(dir, 'sessions.json'),
    JSON.stringify({ schema: 999, appVersion: '0.17.0', gatewayUrl: 'http://studio:9119', data: { total: 1 } })
  )
  assert.equal(cache.readSessions(), null)
})

test('gatewayUrl mismatch discards; appVersion mismatch KEEPS (D5)', () => {
  const dir = tmpDir()
  const writer = makeCache(dir, { gatewayUrl: 'http://other:9119' })
  writer.putSessions({ total: 7 })
  writer.flush()
  // Reader bound to a DIFFERENT gateway: discard.
  const reader = makeCache(dir) // http://studio:9119
  assert.equal(reader.readSessions(), null)

  // Same gateway but different appVersion: KEEP (schema is the contract).
  const writer2 = makeCache(dir, { appVersion: '0.9.9' })
  writer2.putSessions({ total: 3 })
  writer2.flush()
  const reader2 = makeCache(dir, { appVersion: '1.0.0' })
  assert.equal((reader2.readSessions() as any).total, 3)
})

// ------------------------------------------------------------------- eviction

test('cullSession removes the transcript file AND any pending write for it (I4b)', () => {
  const dir = tmpDir()
  const cache = makeCache(dir, { debounceMs: 60_000 })
  cache.putTranscript('dead', [{ text: 'x' }])
  cache.flush() // on disk
  cache.putTranscript('dead', [{ text: 'y' }]) // now ALSO pending again
  cache.cullSession('dead')
  assert.equal(fs.existsSync(path.join(dir, 'transcript-dead.json')), false)
  cache.flush() // the pending write must NOT resurrect the culled file
  assert.equal(fs.existsSync(path.join(dir, 'transcript-dead.json')), false)
})

test('enforceTranscriptCap keeps the newest N by mtime (LRU)', () => {
  const dir = tmpDir()
  const cache = makeCache(dir)
  // 5 transcript files with staggered mtimes
  for (let i = 0; i < 5; i++) {
    const f = path.join(dir, `transcript-s${i}.json`)
    fs.writeFileSync(f, '{}')
    const t = new Date(Date.now() - (5 - i) * 60_000) // s4 newest
    fs.utimesSync(f, t, t)
  }
  cache.enforceTranscriptCap(2)
  const left = fs.readdirSync(dir).sort()
  assert.deepEqual(left, ['transcript-s3.json', 'transcript-s4.json'])
})

test('sweepAgainstLiveSessions culls orphans and reports the count (boot sweep, I4b)', () => {
  const dir = tmpDir()
  const cache = makeCache(dir)
  cache.putTranscript('alive-1', [{}])
  cache.putTranscript('alive-2', [{}])
  cache.putTranscript('ghost-1', [{}])
  cache.putTranscript('ghost-2', [{}])
  cache.flush()
  const culled = cache.sweepAgainstLiveSessions(['alive-1', 'alive-2'])
  assert.equal(culled, 2)
  assert.deepEqual(
    fs.readdirSync(dir).sort(),
    ['transcript-alive-1.json', 'transcript-alive-2.json']
  )
})

test('sweep on an empty/missing dir is a no-op, never throws', () => {
  const cache = makeCache(path.join(tmpDir(), 'never-created'))
  assert.equal(cache.sweepAgainstLiveSessions(['x']), 0)
})

// ---------------------------------------------------------------- misc helpers

test('sessionIdFromTranscriptFile round-trips and rejects non-transcript names', () => {
  assert.equal(sessionIdFromTranscriptFile('transcript-abc.json'), 'abc')
  assert.equal(sessionIdFromTranscriptFile('sessions.json'), null)
  assert.equal(sessionIdFromTranscriptFile('transcript-.json'), null)
})

test('MAX_TRANSCRIPT_FILES default is sane', () => {
  assert.ok(MAX_TRANSCRIPT_FILES >= 50 && MAX_TRANSCRIPT_FILES <= 1000)
})
