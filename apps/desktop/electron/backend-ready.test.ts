/**
 * Tests for electron/backend-ready.ts.
 *
 * Run with: node --test electron/backend-ready.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Covers the cold-start port-announcement deadline (issue #50209): the clock
 * starts before the backend binds its port, so a tight 45s deadline killed a
 * healthy-but-still-compiling backend on cold Windows installs. The default is
 * now cold-start tolerant and overridable via
 * HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS, clamped to a 45s floor.
 */

import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS,
  MIN_PORT_ANNOUNCE_TIMEOUT_MS,
  readDashboardReadyFile,
  readDashboardReadyPayload,
  resolvePortAnnounceTimeoutMs,
  waitForDashboardPort,
  waitForDashboardPortAnnouncement,
  waitForDashboardReadyFile
} from './backend-ready'

type FakeChildProcess = EventEmitter & {
  stdout: EventEmitter
}

// A minimal stand-in for a spawned child process: an EventEmitter with a
// stdout EventEmitter, matching the surface waitForDashboardPort consumes
// (child.stdout.on('data'), child.on('exit'|'error') + the .off() teardown).
function makeFakeChild(): FakeChildProcess {
  const child = new EventEmitter() as FakeChildProcess
  child.stdout = new EventEmitter()

  return child
}

// ---------------------------------------------------------------------------
// resolvePortAnnounceTimeoutMs
// ---------------------------------------------------------------------------

test('default is cold-start tolerant (> the historical 45s floor)', () => {
  assert.equal(resolvePortAnnounceTimeoutMs({}), DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS)
  assert.ok(
    DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS > MIN_PORT_ANNOUNCE_TIMEOUT_MS,
    'cold-start default must exceed the warm-start floor'
  )
})

test('honors a valid HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS override', () => {
  const env = { HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS: '120000' }
  assert.equal(resolvePortAnnounceTimeoutMs(env), 120_000)
})

test('clamps an override below the floor up to the 45s minimum', () => {
  const env = { HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS: '1000' }
  assert.equal(resolvePortAnnounceTimeoutMs(env), MIN_PORT_ANNOUNCE_TIMEOUT_MS)
})

test('rounds a fractional override', () => {
  const env = { HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS: '60000.7' }
  assert.equal(resolvePortAnnounceTimeoutMs(env), 60_001)
})

test('falls back to the default for malformed / non-positive overrides', () => {
  for (const bad of ['', 'abc', '0', '-5', 'NaN', undefined]) {
    const env = bad === undefined ? {} : { HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS: bad }
    assert.equal(
      resolvePortAnnounceTimeoutMs(env),
      DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS,
      `override ${JSON.stringify(bad)} should fall through to the default`
    )
  }
})

// ---------------------------------------------------------------------------
// waitForDashboardPort
// ---------------------------------------------------------------------------

test('resolves with the announced port', async () => {
  const child = makeFakeChild()
  const p = waitForDashboardPort(child, 1000)
  child.stdout.emit('data', 'noise before\nHERMES_DASHBOARD_READY port=54321\n')
  assert.equal(await p, 54321)
})

test('resolves with a HERMES_BACKEND_READY port (headless `serve`)', async () => {
  const child = makeFakeChild()
  const p = waitForDashboardPort(child, 1000)
  child.stdout.emit('data', 'HERMES_BACKEND_READY port=43210\n')
  assert.equal(await p, 43210)
})

test('parses the port even when the line arrives split across chunks', async () => {
  const child = makeFakeChild()
  const p = waitForDashboardPort(child, 1000)
  child.stdout.emit('data', 'HERMES_DASHBOARD_READY po')
  child.stdout.emit('data', 'rt=8080\n')
  assert.equal(await p, 8080)
})

test('rejects when the child exits before announcing', async () => {
  const child = makeFakeChild()
  const p = waitForDashboardPort(child, 1000)
  child.emit('exit', 1, null)
  await assert.rejects(p, /exited before port announcement/)
})

test('rejects on a child error event', async () => {
  const child = makeFakeChild()
  const p = waitForDashboardPort(child, 1000)
  child.emit('error', new Error('spawn ENOENT'))
  await assert.rejects(p, /spawn ENOENT/)
})

test('rejects with the timeout message after the deadline', async () => {
  const child = makeFakeChild()
  await assert.rejects(
    waitForDashboardPort(child, 20),
    /Timed out waiting for Hermes backend port announcement \(20ms\)/
  )
})

test('a late announcement after timeout does not throw (listeners torn down)', async () => {
  const child = makeFakeChild()
  await assert.rejects(waitForDashboardPort(child, 20), /Timed out/)
  // The orphaned backend may still print its READY line later; the watcher
  // must have detached so this emit is a no-op rather than a double-settle.
  assert.doesNotThrow(() => {
    child.stdout.emit('data', 'HERMES_DASHBOARD_READY port=9999\n')
  })
})

// ---------------------------------------------------------------------------
// ready-file port announcement
// ---------------------------------------------------------------------------

function mkTmpReadyFile() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ready-test-'))

  return {
    dir,
    file: path.join(dir, 'ready.json'),
    cleanup: () => fs.rmSync(dir, { recursive: true, force: true })
  }
}

test('readDashboardReadyFile returns a valid port from JSON', () => {
  const tmp = mkTmpReadyFile()

  try {
    fs.writeFileSync(tmp.file, JSON.stringify({ port: 4567 }))
    assert.equal(readDashboardReadyFile(tmp.file), 4567)
  } finally {
    tmp.cleanup()
  }
})

test('readDashboardReadyPayload returns the backend-resolved session token', () => {
  const tmp = mkTmpReadyFile()

  try {
    fs.writeFileSync(tmp.file, JSON.stringify({ port: 4567, session_token: 'resolved-by-backend' }))
    assert.deepEqual(readDashboardReadyPayload(tmp.file), {
      port: 4567,
      sessionToken: 'resolved-by-backend'
    })
  } finally {
    tmp.cleanup()
  }
})

test('readDashboardReadyPayload accepts the old port-only ready-file format', () => {
  const tmp = mkTmpReadyFile()

  try {
    fs.writeFileSync(tmp.file, JSON.stringify({ port: 4567 }))
    assert.deepEqual(readDashboardReadyPayload(tmp.file), { port: 4567, sessionToken: null })
  } finally {
    tmp.cleanup()
  }
})

test('readDashboardReadyFile ignores missing, malformed, or invalid files', () => {
  const tmp = mkTmpReadyFile()

  try {
    assert.equal(readDashboardReadyFile(tmp.file), null)
    fs.writeFileSync(tmp.file, '{')
    assert.equal(readDashboardReadyFile(tmp.file), null)
    fs.writeFileSync(tmp.file, JSON.stringify({ port: 0 }))
    assert.equal(readDashboardReadyFile(tmp.file), null)
  } finally {
    tmp.cleanup()
  }
})

test('waitForDashboardReadyFile resolves when the ready file appears', async () => {
  const tmp = mkTmpReadyFile()
  const child = makeFakeChild()

  try {
    const p = waitForDashboardReadyFile(tmp.file, child, 1000)
    setTimeout(() => fs.writeFileSync(tmp.file, JSON.stringify({ port: 8765 })), 20)
    assert.equal(await p, 8765)
  } finally {
    tmp.cleanup()
  }
})

test('waitForDashboardPortAnnouncement uses ready file when provided', async () => {
  const tmp = mkTmpReadyFile()
  const child = makeFakeChild()

  try {
    const p = waitForDashboardPortAnnouncement(child, { readyFile: tmp.file, timeoutMs: 1000 })
    setTimeout(() => fs.writeFileSync(tmp.file, JSON.stringify({ port: 9876 })), 20)
    assert.equal(await p, 9876)
  } finally {
    tmp.cleanup()
  }
})

test('waitForDashboardPortAnnouncement falls back to stdout for an older backend', async () => {
  const tmp = mkTmpReadyFile()
  const child = makeFakeChild()

  try {
    const p = waitForDashboardPortAnnouncement(child, { readyFile: tmp.file, timeoutMs: 1000 })
    child.stdout.emit('data', 'HERMES_BACKEND_READY port=9876\n')
    assert.equal(await p, 9876)
  } finally {
    tmp.cleanup()
  }
})

test('waitForDashboardReadyFile rejects when the child exits before file readiness', async () => {
  const tmp = mkTmpReadyFile()
  const child = makeFakeChild()

  try {
    const p = waitForDashboardReadyFile(tmp.file, child, 1000)
    child.emit('exit', 1, null)
    await assert.rejects(p, /exited before port announcement/)
  } finally {
    tmp.cleanup()
  }
})

// ---------------------------------------------------------------------------
// describeOutputTail (#93608): the child's real stderr reaches the exit error
// ---------------------------------------------------------------------------

test('exit-before-announcement error carries the buffered output tail (stdout path)', async () => {
  const child = makeFakeChild()

  const wait = waitForDashboardPortAnnouncement(child, {
    describeOutputTail: () => '\nRecent backend output:\nModuleNotFoundError: hermes_cli'
  })

  child.emit('exit', 1, null)

  await assert.rejects(wait, /exited before port announcement \(1\)[\s\S]*ModuleNotFoundError: hermes_cli/)
})

test('exit-before-announcement error carries the buffered output tail (ready-file path)', async () => {
  const child = makeFakeChild()
  const readyFile = path.join(os.tmpdir(), `hermes-ready-${process.pid}-${Date.now()}.json`)

  const wait = waitForDashboardPortAnnouncement(child, {
    describeOutputTail: () => '\nRecent backend output:\nTraceback (most recent call last)',
    readyFile
  })

  child.emit('exit', null, 'SIGSEGV')

  await assert.rejects(wait, /exited before port announcement \(SIGSEGV\)[\s\S]*Traceback/)
})

test('exit-before-announcement error stays clean when no output was buffered', async () => {
  const child = makeFakeChild()

  const wait = waitForDashboardPortAnnouncement(child, {})

  child.emit('exit', 137, null)

  await assert.rejects(wait, error => {
    assert.match((error as Error).message, /exited before port announcement \(137\)$/)

    return true
  })
})

// ---------------------------------------------------------------------------
// bufferedOutput (#60323): a sentinel consumed BEFORE the wait attaches must
// still resolve. main.ts attaches an output tail at spawn, then awaits
// claimBackendChild/advanceBootProgress before calling this wait; flowing-mode
// stdout never replays consumed chunks to late listeners.
// ---------------------------------------------------------------------------

test('resolves from bufferedOutput when the sentinel was consumed before the wait attached (#60323)', async () => {
  const child = makeFakeChild()

  // Simulate the spawn-time output tail: it consumed the READY line already,
  // and no further stdout data will ever arrive.
  const alreadyConsumed = 'boot noise\nHERMES_BACKEND_READY port=43211\n'

  const port = await waitForDashboardPortAnnouncement(child, {
    bufferedOutput: () => alreadyConsumed,
    timeoutMs: 500
  })

  assert.equal(port, 43211)
})

test('bufferedOutput accepts the legacy HERMES_DASHBOARD_READY sentinel too', async () => {
  const child = makeFakeChild()

  const port = await waitForDashboardPortAnnouncement(child, {
    bufferedOutput: () => 'HERMES_DASHBOARD_READY port=43212\n',
    timeoutMs: 500
  })

  assert.equal(port, 43212)
})

test('bufferedOutput without a sentinel still resolves from later live stdout', async () => {
  const child = makeFakeChild()

  const wait = waitForDashboardPortAnnouncement(child, {
    bufferedOutput: () => 'uvicorn still importing...\n',
    timeoutMs: 1000
  })

  child.stdout.emit('data', Buffer.from('HERMES_BACKEND_READY port=43213\n'))

  assert.equal(await wait, 43213)
})

test('bufferedOutput without a sentinel still times out (no false positive)', async () => {
  const child = makeFakeChild()

  const wait = waitForDashboardPort(
    child,
    50,
    () => '',
    () => 'no sentinel here\n'
  )

  await assert.rejects(wait, /Timed out waiting/)
})
