/**
 * Unit tests for the pure window-session-state helpers: sanitize/parse/
 * serialize, and the restore loop's stale-session and quit-abort handling.
 * Mirrors window-state.test.ts's "test the pure logic without booting
 * Electron" pattern.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  parseWindowSessionState,
  restoredSessionWindowOptions,
  restoreSessionWindows,
  sanitizeWindowSessionState,
  type SavedSessionWindow,
  serializeWindowSessionState
} from './window-session-state'

const GEOMETRY = { width: 500, height: 700, isMaximized: false }

function window(sessionId: string, overrides: Partial<SavedSessionWindow> = {}): SavedSessionWindow {
  return { profile: 'default', connectionId: null, sessionId, watch: false, geometry: GEOMETRY, ...overrides }
}

// ─── sanitizeWindowSessionState ─────────────────────────────────────────────

test('sanitizeWindowSessionState rejects garbage/wrong-version input', () => {
  for (const bad of [null, undefined, 'nope', 42, {}, { version: 2, windows: [] }, { version: 1, windows: 'x' }]) {
    assert.equal(sanitizeWindowSessionState(bad), null)
  }
})

test('sanitizeWindowSessionState keeps a valid window and drops a garbage one', () => {
  const result = sanitizeWindowSessionState({
    version: 1,
    windows: [
      { profile: 'default', connectionId: null, sessionId: 'sess-1', watch: true, geometry: GEOMETRY },
      { profile: 'default', connectionId: null, sessionId: '', geometry: GEOMETRY }, // empty sessionId
      { profile: 'default', connectionId: null, geometry: GEOMETRY }, // missing sessionId
      { profile: 'default', connectionId: null, sessionId: 'sess-2', geometry: { width: 'x' } } // bad geometry
    ]
  })

  assert.equal(result?.windows.length, 1)
  assert.equal(result?.windows[0].sessionId, 'sess-1')
  assert.equal(result?.windows[0].watch, true)
})

test('sanitizeWindowSessionState deduplicates by (connectionId, profile, sessionId)', () => {
  const result = sanitizeWindowSessionState({
    version: 1,
    windows: [window('sess-1'), window('sess-1'), window('sess-1', { connectionId: 'conn-a' })]
  })

  assert.equal(result?.windows.length, 2)
})

test('sanitizeWindowSessionState never carries a secret-shaped value through unexamined', () => {
  // profile/connectionId must be plain identifiers; a token/URL-shaped value
  // is still just a string here (the module has no network concept), so the
  // real guarantee under test is that unknown extra fields never survive.
  const result = sanitizeWindowSessionState({
    version: 1,
    windows: [{ ...window('sess-1'), bearerToken: 'sk-should-not-survive', wsUrl: 'wss://leak' }]
  })

  const kept = result?.windows[0] as any
  assert.equal(kept.bearerToken, undefined)
  assert.equal(kept.wsUrl, undefined)
})

// ─── parse / serialize round-trip ───────────────────────────────────────────

test('serializeWindowSessionState / parseWindowSessionState round-trip', () => {
  const windows = [window('sess-1', { connectionId: 'conn-a', watch: true })]
  const parsed = parseWindowSessionState(serializeWindowSessionState(windows))
  assert.deepEqual(parsed?.windows, windows)
})

test('parseWindowSessionState returns null on malformed JSON', () => {
  assert.equal(parseWindowSessionState('{not json'), null)
})

// ─── restoreSessionWindows ───────────────────────────────────────────────────

test('restoreSessionWindows opens every window whose session still exists', async () => {
  const opened: string[] = []
  await restoreSessionWindows(sanitizeWindowSessionState({ version: 1, windows: [window('a'), window('b')] }), {
    exists: async () => true,
    open: w => opened.push(w.sessionId),
    quitting: () => false,
    log: () => {}
  })
  assert.deepEqual(opened, ['a', 'b'])
})

test('restoreSessionWindows skips a stale session without throwing and logs it', async () => {
  const opened: string[] = []
  const logs: string[] = []
  await restoreSessionWindows(sanitizeWindowSessionState({ version: 1, windows: [window('gone'), window('live')] }), {
    exists: async w => w.sessionId === 'live',
    open: w => opened.push(w.sessionId),
    quitting: () => false,
    log: m => logs.push(m)
  })
  assert.deepEqual(opened, ['live'])
  assert.equal(logs.length, 1)
})

test('restoreSessionWindows never throws when the existence probe rejects', async () => {
  const opened: string[] = []
  await assert.doesNotReject(() =>
    restoreSessionWindows(sanitizeWindowSessionState({ version: 1, windows: [window('a'), window('b')] }), {
      exists: async w => {
        if (w.sessionId === 'a') {
          throw new Error('backend unreachable')
        }

        return true
      },
      open: w => opened.push(w.sessionId),
      quitting: () => false,
      log: () => {}
    })
  )
  assert.deepEqual(opened, ['b'])
})

test('restoreSessionWindows stops opening more windows once quitting() flips true', async () => {
  const opened: string[] = []
  let quit = false
  await restoreSessionWindows(sanitizeWindowSessionState({ version: 1, windows: [window('a'), window('b')] }), {
    exists: async () => {
      quit = true // simulate a quit arriving mid-restore

      return true
    },
    open: w => opened.push(w.sessionId),
    quitting: () => quit,
    log: () => {}
  })
  assert.deepEqual(opened, [])
})

// ─── restoredSessionWindowOptions ───────────────────────────────────────────

test('restoredSessionWindowOptions clamps to the available display', () => {
  const small = [{ workArea: { x: 0, y: 0, width: 600, height: 700 } }]
  const oversized = window('a', { geometry: { width: 5000, height: 5000, isMaximized: false } }).geometry
  const opts = restoredSessionWindowOptions({ ...window('a'), geometry: oversized }, small)
  assert.ok(opts.width <= 600)
  assert.ok(opts.height <= 700)
})
