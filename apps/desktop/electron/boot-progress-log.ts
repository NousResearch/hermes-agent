/**
 * desktop.log lines for boot-progress messages, one per change.
 *
 * Boot progress is re-published on a timer while main waits: a launch parked
 * behind a live update ticks "An update is finishing — …" every second for
 * the whole update (#125612). The renderer needs every tick; the log does not.
 * Logging each one filled the 300-line in-memory tail — the lines boot-failure
 * and connection reports attach — with one repeated sentence, so a real
 * failure's report carried no context. Log a message when it differs from the
 * previous one; a phase that recurs after something else is logged again.
 */
export function createBootMessageLogger(log: (line: string) => void) {
  let previous: null | string = null

  return (message: string | undefined) => {
    if (!message || message === previous) {
      return
    }

    previous = message
    log(`[boot] ${message}`)
  }
}
