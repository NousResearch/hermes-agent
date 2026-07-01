const fs = require('node:fs')
const path = require('node:path')

const _READY_RE = /^HERMES_DASHBOARD_READY port=(\d+)/m

// The announcement clock starts the instant the backend process is spawned —
// before uvicorn binds its socket. On a cold install the child must first
// compile and import the whole `hermes_cli.main` → `web_server` → FastAPI/
// uvicorn chain, and on Windows real-time AV (Defender) scans every freshly
// written `.pyc`. That pre-bind cost can run 30-60s on a slow disk, so a tight
// 45s deadline kills a *healthy but still-starting* backend and respawns it,
// piling up orphaned processes (issue #50209). A roomier default absorbs the
// cold-start cost; a warm start still announces in well under a second.
const DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS = 90_000
// Never trust a deadline tighter than the warm-start path needs; floor at 45s
// (the historical default) so a malformed override can't reintroduce the loop.
const MIN_PORT_ANNOUNCE_TIMEOUT_MS = 45_000

/**
 * Resolve the port-announcement deadline. Honors the
 * HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS env override (for users on slow
 * disks / aggressive AV who need an even longer cold-start window), clamped
 * to a sane floor so a bad value can't make boot flakier than the default.
 */
function resolvePortAnnounceTimeoutMs(env = process.env) {
  const parsed = Number(env.HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS)
  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.max(MIN_PORT_ANNOUNCE_TIMEOUT_MS, Math.round(parsed))
  }
  return DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS
}

/**
 * Watch a child process's stdout for the `HERMES_DASHBOARD_READY port=<N>`
 * line that web_server.py prints after uvicorn binds its socket.
 *
 * Returns the parsed port. Rejects if:
 *   - the child exits before emitting the line
 *   - the child emits an `error` event
 *   - no line arrives within the timeout
 *
 * The default timeout is cold-start tolerant (see
 * DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS) because the clock starts before the
 * backend has even bound its port. Pass an explicit `timeoutMs` to override.
 *
 * A single `cleanup()` tears down every listener (data/exit/error/timeout)
 * on every terminal path — resolve, reject, or timeout — so repeated
 * backend spawns don't leak listener slots on the child.
 */
function waitForDashboardPort(child, timeoutMs = resolvePortAnnounceTimeoutMs()) {
  return new Promise((resolve, reject) => {
    let buf = ''
    let done = false

    function cleanup() {
      if (done) return
      done = true
      clearTimeout(timer)
      child.stdout.off('data', onData)
      child.off('exit', onExit)
      child.off('error', onError)
    }

    function onData(chunk) {
      buf += chunk.toString()
      let nl
      while ((nl = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, nl)
        buf = buf.slice(nl + 1)
        const m = line.match(_READY_RE)
        if (m) {
          cleanup()
          resolve(parseInt(m[1], 10))
          return
        }
      }
    }

    function onExit(code, signal) {
      cleanup()
      reject(new Error(`Hermes backend: exited before port announcement (${signal || code})`))
    }

    function onError(err) {
      cleanup()
      reject(err)
    }

    const timer = setTimeout(() => {
      cleanup()
      reject(new Error(`Timed out waiting for Hermes backend port announcement (${timeoutMs}ms)`))
    }, timeoutMs)

    child.stdout.on('data', onData)
    child.on('exit', onExit)
    child.on('error', onError)
  })
}

function readDashboardReadyFile(readyFile) {
  if (!readyFile) return null
  try {
    const parsed = JSON.parse(fs.readFileSync(readyFile, 'utf8'))
    const port = Number(parsed?.port)
    return Number.isInteger(port) && port > 0 ? port : null
  } catch {
    return null
  }
}

// readDashboardReadyInfo — like readDashboardReadyFile but returns the full {port, projectRoot} record.
// project_root is the tree the backend is actually running from (SPEC 2026-07-01 AC-3 effect gate); a
// pre-migration backend that never emits it yields projectRoot=null.
function readDashboardReadyInfo(readyFile) {
  if (!readyFile) return null
  try {
    const parsed = JSON.parse(fs.readFileSync(readyFile, 'utf8'))
    const port = Number(parsed?.port)
    if (!(Number.isInteger(port) && port > 0)) return null
    const projectRoot = typeof parsed?.project_root === 'string' && parsed.project_root ? parsed.project_root : null
    return { port, projectRoot }
  } catch {
    return null
  }
}

// assertBackendProjectRoot — AC-3 EFFECT gate. True iff the backend's emitted project_root is UNDER
// expectedRoot. FAIL-CLOSED: a missing/empty project_root (an un-migrated/older backend) returns false —
// it must NOT pass by omission (that would resurrect the proxy-gate problem the effect gate exists to kill).
function assertBackendProjectRoot(readyInfo, expectedRoot, { pathModule = path } = {}) {
  if (!readyInfo || !readyInfo.projectRoot || !expectedRoot) return false
  const actual = pathModule.resolve(readyInfo.projectRoot)
  const expected = pathModule.resolve(expectedRoot)
  if (actual === expected) return true
  const rel = pathModule.relative(expected, actual)
  return rel !== '' && !rel.startsWith('..') && !pathModule.isAbsolute(rel)
}

function waitForDashboardReadyFile(readyFile, child, timeoutMs = resolvePortAnnounceTimeoutMs()) {
  return new Promise((resolve, reject) => {
    let done = false
    let interval = null

    function cleanup() {
      if (done) return
      done = true
      clearTimeout(timer)
      if (interval) clearInterval(interval)
      child.off('exit', onExit)
      child.off('error', onError)
    }

    function check() {
      const port = readDashboardReadyFile(readyFile)
      if (port) {
        cleanup()
        resolve(port)
      }
    }

    function onExit(code, signal) {
      cleanup()
      reject(new Error(`Hermes backend: exited before port announcement (${signal || code})`))
    }

    function onError(err) {
      cleanup()
      reject(err)
    }

    const timer = setTimeout(() => {
      cleanup()
      reject(new Error(`Timed out waiting for Hermes backend port announcement (${timeoutMs}ms)`))
    }, timeoutMs)

    child.on('exit', onExit)
    child.on('error', onError)
    interval = setInterval(check, 50)
    if (typeof interval.unref === 'function') interval.unref()
    check()
  })
}

function waitForDashboardPortAnnouncement(child, options = {}) {
  const timeoutMs = options.timeoutMs ?? resolvePortAnnounceTimeoutMs()
  if (options.readyFile) {
    return waitForDashboardReadyFile(options.readyFile, child, timeoutMs)
  }
  return waitForDashboardPort(child, timeoutMs)
}

module.exports = {
  waitForDashboardPort,
  waitForDashboardPortAnnouncement,
  waitForDashboardReadyFile,
  readDashboardReadyFile,
  readDashboardReadyInfo,
  assertBackendProjectRoot,
  resolvePortAnnounceTimeoutMs,
  DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS,
  MIN_PORT_ANNOUNCE_TIMEOUT_MS
}
