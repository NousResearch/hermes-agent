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
  restoredInstanceWindowOptions,
  restoredSessionWindowOptions,
  restoreInstanceWindows,
  restoreSessionWindows,
  sanitizeWindowSessionState,
  type SavedInstanceWindow,
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

// ─── instance windows (File > New Window) ───────────────────────────────────

function instance(overrides: Partial<SavedInstanceWindow> = {}): SavedInstanceWindow {
  return { profile: 'default', connectionId: null, geometry: GEOMETRY, windowId: null, profileWindow: false, ...overrides }
}

test('sanitizeWindowSessionState keeps valid instance windows and drops garbage ones', () => {
  const result = sanitizeWindowSessionState({
    version: 1,
    windows: [],
    instanceWindows: [
      { profile: 'default', connectionId: null, geometry: GEOMETRY },
      { profile: 'work', connectionId: 'conn-a', geometry: GEOMETRY },
      { profile: 'default', connectionId: null, geometry: { width: 'x' } }, // bad geometry
      null // garbage row
    ]
  })

  assert.equal(result?.instanceWindows.length, 2)
  assert.equal(result?.instanceWindows[0].profile, 'default')
  assert.equal(result?.instanceWindows[1].connectionId, 'conn-a')
})

// --- Stable per-window identity (BUILD-STABLE-WINDOW-ID) --------------------
//
// RED: createInstanceWindow always mints a fresh crypto.randomUUID() and
// SavedInstanceWindow/sanitizeWindowSessionState never persisted a window's
// id, so a restored peer could never recover its own scoped
// sessionTiles.v2/layoutTree.v2 storage (windowScopedKey in src/store/windows.ts
// namespaces by that id). These tests fail against the pre-fix shape (no
// `windowId` field) and the pre-fix (connectionId, profile) dedup, which
// collapsed multiple same-route peers into one saved row.

test('sanitizeWindowSessionState preserves multiple instance windows sharing the same profile/connection (distinct windowId)', () => {
  const result = sanitizeWindowSessionState({
    version: 1,
    windows: [],
    instanceWindows: [
      instance({ windowId: 'win-a' }),
      instance({ windowId: 'win-b' }),
      instance({ windowId: 'win-c' })
    ]
  })

  assert.equal(result?.instanceWindows.length, 3)
  assert.deepEqual(
    result?.instanceWindows.map(w => w.windowId).sort(),
    ['win-a', 'win-b', 'win-c']
  )
})

test('sanitizeWindowSessionState drops a second instance window row that collides on the exact same windowId', () => {
  const result = sanitizeWindowSessionState({
    version: 1,
    windows: [],
    instanceWindows: [instance({ windowId: 'dup-1' }), instance({ windowId: 'dup-1', profile: 'work' })]
  })

  assert.equal(result?.instanceWindows.length, 1)
  assert.equal(result?.instanceWindows[0].profile, 'default')
})

test('sanitizeWindowSessionState rejects a malformed windowId (degrades to null, never propagates garbage)', () => {
  const result = sanitizeWindowSessionState({
    version: 1,
    windows: [],
    instanceWindows: [
      instance({ windowId: 'has space' }),
      instance({ windowId: '../etc/passwd' }),
      instance({ windowId: 'a'.repeat(65) }),
      instance({ windowId: 42 as unknown as string })
    ]
  })

  assert.equal(result?.instanceWindows.length, 4)
  assert.ok(result?.instanceWindows.every(w => w.windowId === null))
})

test('sanitizeWindowSessionState keeps legacy instance windows (no windowId at all) without colliding on each other', () => {
  const result = sanitizeWindowSessionState({
    version: 1,
    windows: [],
    instanceWindows: [instance(), instance(), instance({ connectionId: 'conn-a' })]
  })

  // Pre-fix behavior deduped these down to 2 by (connectionId, profile); a
  // legacy record carries no identity to dedupe by, so every one must survive.
  assert.equal(result?.instanceWindows.length, 3)
  assert.ok(result?.instanceWindows.every(w => w.windowId === null))
})

test('serializeWindowSessionState / parseWindowSessionState round-trip preserves windowId', () => {
  const windows: SavedSessionWindow[] = []
  const instances = [instance({ connectionId: 'conn-a', windowId: 'win-a' }), instance({ windowId: null })]
  const parsed = parseWindowSessionState(serializeWindowSessionState(windows, instances))
  assert.deepEqual(parsed?.instanceWindows, instances)
})

test('profileWindow preserves explicit booleans and defaults missing or malformed legacy metadata to unpinned', () => {
  const rawPins = [true, false, undefined, 'true', 1, null]

  const parsed = parseWindowSessionState(JSON.stringify({ version: 1, windows: [], instanceWindows:
    rawPins.map((profileWindow, index) => ({ ...instance({ windowId: `peer-${index}` }), profileWindow }))
  }))

  assert.deepEqual(parsed?.instanceWindows.map(w => w.profileWindow), [true, false, false, false, false, false])
  assert.deepEqual(parseWindowSessionState(serializeWindowSessionState([], parsed!.instanceWindows)), parsed)
})

test('sanitizeWindowSessionState defaults instanceWindows to [] when the field is absent (back-compat with pre-fix saved state)', () => {
  const result = sanitizeWindowSessionState({ version: 1, windows: [window('sess-1')] })
  assert.deepEqual(result?.instanceWindows, [])
  assert.equal(result?.windows.length, 1)
})

test('serializeWindowSessionState / parseWindowSessionState round-trip includes instance windows', () => {
  const windows = [window('sess-1')]
  const instances = [instance({ connectionId: 'conn-a' })]
  const parsed = parseWindowSessionState(serializeWindowSessionState(windows, instances))
  assert.deepEqual(parsed?.windows, windows)
  assert.deepEqual(parsed?.instanceWindows, instances)
})

test('restoreInstanceWindows opens every saved instance window unconditionally (no existence probe)', () => {
  const opened: Array<string | null> = []
  restoreInstanceWindows(sanitizeWindowSessionState({ version: 1, windows: [], instanceWindows: [instance(), instance({ profile: 'work' })] }), {
    open: w => opened.push(w.profile),
    quitting: () => false
  })
  assert.deepEqual(opened, ['default', 'work'])
})

test('restoreInstanceWindows stops opening more windows once quitting() flips true', () => {
  const opened: Array<string | null> = []
  let quit = false
  restoreInstanceWindows(sanitizeWindowSessionState({ version: 1, windows: [], instanceWindows: [instance(), instance({ profile: 'work' })] }), {
    open: w => {
      quit = true
      opened.push(w.profile)
    },
    quitting: () => quit
  })
  assert.deepEqual(opened, ['default'])
})

test('restoredInstanceWindowOptions clamps to the available display', () => {
  const small = [{ workArea: { x: 0, y: 0, width: 600, height: 700 } }]
  const oversized = instance({ geometry: { width: 5000, height: 5000, isMaximized: false } }).geometry
  const opts = restoredInstanceWindowOptions({ ...instance(), geometry: oversized }, small)
  assert.ok(opts.width <= 600)
  assert.ok(opts.height <= 700)
})
