const _READY_RE = /^HERMES_DASHBOARD_READY port=(\d+)/m

// Default boot timeout for the HERMES_DASHBOARD_READY announcement.
// Cold starts on machines with a large SQLite DB or many installed skills can
// take well over a minute before uvicorn binds and emits the ready line.
// 120 s keeps things responsive while allowing room for slow first-boot scans.
// Override with HERMES_DESKTOP_BOOT_TIMEOUT_MS (milliseconds) if needed.
const DEFAULT_BOOT_TIMEOUT_MS = Number(process.env.HERMES_DESKTOP_BOOT_TIMEOUT_MS) || 120_000

/**
 * Watch a child process's stdout for the `HERMES_DASHBOARD_READY port=<N>`
 * line that web_server.py prints after uvicorn binds its socket.
 *
 * Returns the parsed port. Rejects if:
 *   - the child exits before emitting the line
 *   - the child emits an `error` event
 *   - no line arrives within the timeout
 *
 * A single `cleanup()` tears down every listener (data/exit/error/timeout)
 * on every terminal path — resolve, reject, or timeout — so repeated
 * backend spawns don't leak listener slots on the child.
 */
function waitForDashboardPort(child, timeoutMs = DEFAULT_BOOT_TIMEOUT_MS) {
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

module.exports = { waitForDashboardPort }
