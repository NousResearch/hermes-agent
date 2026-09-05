import fs from 'node:fs'

// `hermes serve` announces HERMES_BACKEND_READY; the legacy `hermes dashboard`
// backend announces HERMES_DASHBOARD_READY. Accept either so the desktop spawn
// works against both the headless backend and old/dashboard runtimes.
const _READY_RE = /^HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)/m

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
 * Watch a child process's stdout for the `HERMES_(BACKEND|DASHBOARD)_READY
 * port=<N>` line that web_server.py prints after uvicorn binds its socket.
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
function waitForDashboardPort(
  child,
  timeoutMs = resolvePortAnnounceTimeoutMs(),
  describeOutputTail = () => '',
  bufferedOutput: () => string = () => ''
) {
  return new Promise((resolve, reject) => {
    // Seed the line buffer with any output the spawn-time tail already
    // consumed (#60323): main.ts attaches its output tail at spawn, then
    // awaits claimBackendChild + advanceBootProgress BEFORE this listener
    // attaches. child.stdout is in flowing mode from the tail's listener, so
    // a READY line flushed during that window is emitted once and never
    // replayed to late listeners — the wait then times out at 90s and a
    // healthy backend is killed. Scanning the tail's buffer (and seeding any
    // trailing partial line) makes the listener-attach ordering irrelevant.
    let buf = ''
    let done = false

    function cleanup() {
      if (done) {
        return
      }

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
      reject(new Error(`Hermes backend: exited before port announcement (${signal || code})${describeOutputTail()}`))
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

    // Listener is live — now recover a sentinel that was already flushed and
    // consumed before this promise existed. The snapshot is taken AFTER the
    // listener attaches, so no chunk can fall between snapshot and listener.
    if (!done) {
      const alreadyBuffered = bufferedOutput()
      const m = alreadyBuffered ? alreadyBuffered.match(_READY_RE) : null

      if (m) {
        cleanup()
        resolve(parseInt(m[1], 10))
      }
    }
  })
}

function readDashboardReadyFile(readyFile: fs.PathOrFileDescriptor) {
  return readDashboardReadyPayload(readyFile)?.port ?? null
}

/**
 * Read the private local backend handshake. The token is optional so a new
 * Desktop remains compatible with an older backend that wrote only {port}.
 */
function readDashboardReadyPayload(readyFile: fs.PathOrFileDescriptor) {
  if (!readyFile) {
    return null
  }

  try {
    const parsed = JSON.parse(fs.readFileSync(readyFile, 'utf8'))
    const port = Number(parsed?.port)

    if (!Number.isInteger(port) || port <= 0) {
      return null
    }

    return {
      port,
      sessionToken: typeof parsed?.session_token === 'string' && parsed.session_token ? parsed.session_token : null
    }
  } catch {
    return null
  }
}

function waitForDashboardReadyFile(
  readyFile,
  child,
  timeoutMs = resolvePortAnnounceTimeoutMs(),
  describeOutputTail = () => ''
) {
  return new Promise((resolve, reject) => {
    let done = false
    let interval = null

    function cleanup() {
      if (done) {
        return
      }

      done = true
      clearTimeout(timer)

      if (interval) {
        clearInterval(interval)
      }

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
      reject(new Error(`Hermes backend: exited before port announcement (${signal || code})${describeOutputTail()}`))
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

    if (typeof interval.unref === 'function') {
      interval.unref()
    }

    check()
  })
}

function waitForDashboardPortAnnouncement(
  child,
  options: {
    /**
     * Returns the child's output buffered since SPAWN (the output tail's
     * accumulated text, #60323). Scanned for an already-emitted READY
     * sentinel so attaching this wait AFTER other awaits (backend claim,
     * boot-progress IPC) can never lose the announcement: flowing-mode
     * stdout never replays chunks to late listeners.
     */
    bufferedOutput?: () => string
    /** Returns a formatted stdout/stderr tail suffix for exit errors (#93608). */
    describeOutputTail?: () => string
    readyFile?: fs.PathOrFileDescriptor | null
    timeoutMs?: number
  } = {}
) {
  const timeoutMs = options.timeoutMs ?? resolvePortAnnounceTimeoutMs()
  const describeOutputTail = options.describeOutputTail ?? (() => '')

  if (!options.readyFile) {
    return waitForDashboardPort(child, timeoutMs, describeOutputTail, options.bufferedOutput ?? (() => ''))
  }

  // New backends publish a private ready file; older runtimes ignore its env
  // var and only emit the stdout sentinel. Watch both through one lifecycle so
  // either compatibility path wins without leaving a timer/listener behind.
  return new Promise((resolve, reject) => {
    let done = false
    let interval = null
    let buf = ''

    function cleanup() {
      if (done) {
        return
      }

      done = true
      clearTimeout(timer)

      if (interval) {
        clearInterval(interval)
      }
      child.stdout.off('data', onData)
      child.off('exit', onExit)
      child.off('error', onError)
    }

    function resolvePort(port) {
      cleanup()
      resolve(port)
    }

    function checkReadyFile() {
      const payload = readDashboardReadyPayload(options.readyFile)

      if (payload) {
        resolvePort(payload.port)
      }
    }

    function onData(chunk) {
      buf += chunk.toString()
      let nl

      while ((nl = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, nl)
        buf = buf.slice(nl + 1)
        const match = line.match(_READY_RE)

        if (match) {
          resolvePort(parseInt(match[1], 10))

          return
        }
      }
    }

    function onExit(code, signal) {
      cleanup()
      reject(new Error(`Hermes backend: exited before port announcement (${signal || code})${describeOutputTail()}`))
    }

    function onError(error) {
      cleanup()
      reject(error)
    }

    const timer = setTimeout(() => {
      cleanup()
      reject(new Error(`Timed out waiting for Hermes backend port announcement (${timeoutMs}ms)`))
    }, timeoutMs)

    child.stdout.on('data', onData)
    child.on('exit', onExit)
    child.on('error', onError)
    interval = setInterval(checkReadyFile, 50)

    if (typeof interval.unref === 'function') {
      interval.unref()
    }

    checkReadyFile()

    if (!done) {
      const buffered = options.bufferedOutput?.()
      const match = buffered ? buffered.match(_READY_RE) : null

      if (match) {
        resolvePort(parseInt(match[1], 10))
      }
    }
  })
}

export {
  DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS,
  MIN_PORT_ANNOUNCE_TIMEOUT_MS,
  readDashboardReadyFile,
  readDashboardReadyPayload,
  resolvePortAnnounceTimeoutMs,
  waitForDashboardPort,
  waitForDashboardPortAnnouncement,
  waitForDashboardReadyFile
}
