import fs from 'node:fs'

// `hermes serve` announces HERMES_BACKEND_READY; the legacy `hermes dashboard`
// backend announces HERMES_DASHBOARD_READY. Accept either so the desktop spawn
// works against both the headless backend and old/dashboard runtimes.
const _READY_RE = /^HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)/m

// Same sentinel inside a MERGED stdout+stderr buffer (the spawn-time output tail, a remote
// `>> log 2>&1` file): uvicorn's stderr chunks end without a newline, so the sentinel can be
// spliced onto them (`...process [4711]HERMES_BACKEND_READY port=65238`) and `^` never lines up
// (#103792). Match on a token boundary instead; `port=<digits>` keeps prose mentions out.
export const READY_IN_MERGED_OUTPUT_RE = /(?<!\w)HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)/

// The announcement clock starts the instant the backend process is spawned —
// before uvicorn binds its socket. On a cold install the child must first
// compile and import the whole `hermes_cli.main` → `web_server` → FastAPI/
// uvicorn chain, and on Windows real-time AV (Defender) scans every freshly
// written `.pyc`. That pre-bind cost can run 30-60s on a slow disk, so a tight
// 45s deadline kills a *healthy but still-starting* backend and respawns it,
// piling up orphaned processes (issue #50209). A roomier default absorbs the
// cold-start cost; a warm start still announces in well under a second.
const DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS = 180_000
// Never trust a deadline tighter than the warm-start path needs; floor at 45s
// (the historical default) so a malformed override can't reintroduce the loop.
const MIN_PORT_ANNOUNCE_TIMEOUT_MS = 45_000

// venv_sync emits these banners on stderr before running an owed source-update
// phase. Dependency completion can be quiet for 19-22 minutes. Only this
// explicit phase may use the existing absolute 30-minute budget from spawn;
// repeated banners never move the deadline. Ordinary startup keeps its budget.
const SOURCE_COMPLETION_BANNER_RE =
  /hermes: (?:finishing an interrupted source update|completing source-update dependencies)\.\.\./

const SOURCE_COMPLETION_MAX_TOTAL_MS = 30 * 60_000

// A monotonic clock keeps system-clock corrections from extending startup.
function realNow() {
  return Number(process.hrtime.bigint() / 1_000_000n)
}

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
  bufferedOutput: () => string = () => '',
  readyFile: fs.PathOrFileDescriptor | null = null
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
    let progressBuf = ''
    let done = false
    let readyFileInterval = null
    // #122206: the child is finishing an owed source-update completion
    // (venv_sync banners) before `hermes serve` starts. While that repair is
    // explicitly in progress the startup deadline becomes the absolute phase cap,
    // instead of killing a healthy repair mid-run.
    const startedAt = realNow()
    let completionInProgress = false
    let timer

    function completionDeadline() {
      return completionInProgress ? startedAt + SOURCE_COMPLETION_MAX_TOTAL_MS : null
    }

    function rearmTimer() {
      clearTimeout(timer)
      const deadline = completionDeadline()
      timer = setTimeout(
        () => {
          cleanup()
          reject(
            new Error(
              `Timed out waiting for Hermes backend port announcement (${timeoutMs}ms)${deadline ? ' while an update completion was in progress' : ''}`
            )
          )
        },
        Math.max(0, (deadline ?? startedAt + timeoutMs) - realNow())
      )
    }

    function cleanup() {
      if (done) {
        return
      }

      done = true
      clearTimeout(timer)

      if (readyFileInterval) {
        clearInterval(readyFileInterval)
      }

      child.stdout.off('data', onData)
      child.stderr?.off('data', onProgressData)
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

        // venv_sync's banner is the signal that an owed source-update tail is
        // running ahead of `hermes serve` (#122206): re-arm the deadline so a
        // healthy multi-minute repair is not killed and re-run per boot.
        if (SOURCE_COMPLETION_BANNER_RE.test(line)) {
          completionInProgress = true
          rearmTimer()
        }
      }
    }

    function onProgressData(chunk) {
      // Progress is separate from readiness: a READY-shaped stderr line must
      // never resolve the port. Retain a bounded suffix for split banners.
      const progress = progressBuf + chunk.toString()
      progressBuf = progress.slice(-512)

      if (!completionInProgress && SOURCE_COMPLETION_BANNER_RE.test(progress)) {
        completionInProgress = true
        rearmTimer()
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

    function checkReadyFile() {
      if (!readyFile) {
        return
      }

      const port = readDashboardReadyFile(readyFile)

      if (port) {
        cleanup()
        resolve(port)
      }
    }

    rearmTimer()
    child.stdout.on('data', onData)
    child.stderr?.on('data', onProgressData)
    child.on('exit', onExit)
    child.on('error', onError)

    // Listener is live — now recover a sentinel that was already flushed and
    // consumed before this promise existed. The snapshot is taken AFTER the
    // listener attaches, so no chunk can fall between snapshot and listener.
    // Merged-buffer regex here (the tail interleaves both streams). Currently dormant: both
    // main.ts callers attach the tail and build this wait in one synchronous block, so the
    // snapshot is empty; any await reintroduced between them makes this the live path again.
    if (!done) {
      const alreadyBuffered = bufferedOutput()
      // Preserve a partial producer banner across the snapshot/live boundary.
      progressBuf = alreadyBuffered.slice(-512)
      const m = alreadyBuffered ? alreadyBuffered.match(READY_IN_MERGED_OUTPUT_RE) : null

      if (m) {
        cleanup()
        resolve(parseInt(m[1], 10))
      } else if (alreadyBuffered && SOURCE_COMPLETION_BANNER_RE.test(alreadyBuffered)) {
        // The repair started before this listener attached (the dormant-gap
        // path above): seed the banner so the deadline re-arms like a live one.
        completionInProgress = true
        rearmTimer()
      }
    }

    if (!done && readyFile) {
      readyFileInterval = setInterval(checkReadyFile, 50)

      if (typeof readyFileInterval.unref === 'function') {
        readyFileInterval.unref()
      }

      checkReadyFile()
    }
  })
}

function readDashboardReadyFile(readyFile: fs.PathOrFileDescriptor) {
  if (!readyFile) {
    return null
  }

  try {
    const parsed = JSON.parse(fs.readFileSync(readyFile, 'utf8'))
    const port = Number(parsed?.port)

    return Number.isInteger(port) && port > 0 ? port : null
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

  return waitForDashboardPort(
    child,
    timeoutMs,
    describeOutputTail,
    options.bufferedOutput ?? (() => ''),
    options.readyFile ?? null
  )
}

export {
  DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS,
  MIN_PORT_ANNOUNCE_TIMEOUT_MS,
  readDashboardReadyFile,
  resolvePortAnnounceTimeoutMs,
  waitForDashboardPort,
  waitForDashboardPortAnnouncement,
  waitForDashboardReadyFile
}
