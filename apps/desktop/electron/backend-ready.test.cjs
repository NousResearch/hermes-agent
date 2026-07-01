/**
 * Tests for electron/backend-ready.cjs.
 *
 * Run with: node --test electron/backend-ready.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Covers the cold-start port-announcement deadline (issue #50209): the clock
 * starts before the backend binds its port, so a tight 45s deadline killed a
 * healthy-but-still-compiling backend on cold Windows installs. The default is
 * now cold-start tolerant and overridable via
 * HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS, clamped to a 45s floor.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const { EventEmitter } = require('node:events')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const {
  readDashboardReadyFile,
  readDashboardReadyInfo,
  assertBackendProjectRoot,
  waitForDashboardPort,
  waitForDashboardPortAnnouncement,
  waitForDashboardReadyFile,
  resolvePortAnnounceTimeoutMs,
  DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS,
  MIN_PORT_ANNOUNCE_TIMEOUT_MS
} = require('./backend-ready.cjs')

// A minimal stand-in for a spawned child process: an EventEmitter with a
// stdout EventEmitter, matching the surface waitForDashboardPort consumes
// (child.stdout.on('data'), child.on('exit'|'error') + the .off() teardown).
function makeFakeChild() {
  const child = new EventEmitter()
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

test('readDashboardReadyInfo returns port + projectRoot (AC-3 mechanism)', () => {
  const tmp = mkTmpReadyFile()
  try {
    fs.writeFileSync(tmp.file, JSON.stringify({ port: 4567, project_root: '/Users/x/.hermes/runtime/hermes-agent' }))
    const info = readDashboardReadyInfo(tmp.file)
    assert.equal(info.port, 4567)
    assert.equal(info.projectRoot, '/Users/x/.hermes/runtime/hermes-agent')
    // pre-migration backend (no project_root) -> projectRoot null, port still read
    fs.writeFileSync(tmp.file, JSON.stringify({ port: 4567 }))
    assert.equal(readDashboardReadyInfo(tmp.file).projectRoot, null)
  } finally {
    tmp.cleanup()
  }
})

test('assertBackendProjectRoot: AC-3 effect gate, fail-closed on missing', () => {
  const RT = '/Users/x/.hermes/runtime/hermes-agent'
  // under the runtime tree -> pass
  assert.equal(assertBackendProjectRoot({ port: 1, projectRoot: RT }, RT), true)
  // a subdir under it -> pass
  assert.equal(assertBackendProjectRoot({ port: 1, projectRoot: RT + '/sub' }, RT), true)
  // the DEV tree (sibling) -> fail
  assert.equal(assertBackendProjectRoot({ port: 1, projectRoot: '/Users/x/.hermes/hermes-agent' }, RT), false)
  // MISSING project_root -> FAIL CLOSED (must not pass by omission)
  assert.equal(assertBackendProjectRoot({ port: 1, projectRoot: null }, RT), false)
  assert.equal(assertBackendProjectRoot(null, RT), false)
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
