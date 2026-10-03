import { spawn } from 'node:child_process'

import { hiddenWindowsChildOptions } from '../windows-child-options'

interface StateDbPreflight {
  python: string | null
  script: string
  home: string
  log: (message: string) => void
  /**
   * The installation launcher of a PM-managed checkout (`.hermes/bin/hermes`).
   * A managed checkout carries no venv of its own — the launcher owns
   * interpreter and generation selection there — so the snapshot runs through
   * it exactly like the update check does (`readSourceUpdate`).
   */
  launcher?: string | null
  /**
   * Cap for the snapshot subprocess BEFORE it reports progress. The runtime's
   * snapshot helper (`backup_sqlite.py`) now throttles `PRFL-HB <detail>`
   * stderr lines while a phase is alive; an older runtime prints none, so
   * this window stays the whole budget for it. It replaces the previous flat
   * 180 s cap, which sat ~6% above a healthy 14.40 GB copy measured on
   * Windows (#124972, #124983): a progressing copy now outlives it via
   * heartbeats while a wedged one is still killed.
   */
  timeoutMs?: number
  /**
   * Silence window AFTER the first heartbeat: no `PRFL-HB` line for this long
   * means the copy is wedged (a slow-but-progressing one keeps heartbeating),
   * and the subprocess is killed and the update cancelled.
   */
  stallMs?: number
}

/** Heartbeat prefix the runtime's `backup_sqlite.py` throttles while a phase makes progress. */
const PROGRESS_PREFIX = 'PRFL-HB '
/** No heartbeat may ever arrive before the process has even started: a generous fixed window. */
const DEFAULT_STARTUP_MS = 180_000
/** Default no-progress window once heartbeats have begun. */
const DEFAULT_STALL_MS = 45_000

/** SIGTERM first, then SIGKILL after a grace period. On Windows every signal is
 * TerminateProcess, so the runtime's own watchdog is the kill path that
 * matters there; this one is the belt-and-braces backstop. */
function terminate(child: ReturnType<typeof spawn>): void {
  child.kill('SIGTERM')

  setTimeout((): void => {
    if (child.exitCode === null && child.signalCode === null) {
      child.kill('SIGKILL')
    }
  }, 2_000).unref()
}

function cancelled(message: string, log: (message: string) => void): Error {
  const full: string =
    `${message}. ` +
    'The snapshot could not complete in time (a large or busy state.db — this is not evidence of corruption). ' +
    'Update cancelled before backend shutdown. Update the selected installation with its hermes update command, then retry.'

  log(`[updates] ${full}`)
  const error = new Error(full)

  return error
}

/**
 * Snapshot `state.db` before the backend stops.
 *
 * Async by design: the caller's ordering constraint is that the snapshot
 * finishes before `stopBackendsForUpdate` — which it awaits — not that the
 * Electron main thread blocks on a synchronous spawn. The synchronous form
 * wedged the whole main process (windows, IPC, renderer readiness) for the
 * full probe duration and surfaced an opaque `spawnSync ... ETIMEDOUT`
 * (#124972, #103786). The snapshot still runs to completion or is cancelled
 * before the backend shutdown: nothing downstream observes a half-run probe.
 *
 * The child is killed only when it stops proving progress: the runtime's
 * snapshot helper heartbeats on stderr while copying or quick-checking, so a
 * legitimately slow multi-GB snapshot is never cut down by a fixed wall-clock
 * cap again (#124983).
 */
export async function preflightStateDb({
  python,
  script,
  home,
  log,
  launcher = null,
  timeoutMs = DEFAULT_STARTUP_MS,
  stallMs = DEFAULT_STALL_MS
}: StateDbPreflight): Promise<void> {
  let command: string | null = launcher ?? python

  if (!command) {
    const message = 'Python not found'

    log(`[updates] state.db pre-flight failed: ${message}. Update cancelled before backend shutdown.`)
    throw new Error(message)
  }

  let args: string[] = launcher
    ? ['--run-module', 'hermes_cli.backup_sqlite', home]
    : ['-I', '-S', script, home]

  // Node refuses direct .cmd execFile; an older published launcher can still
  // be one. Same fail-closed guard as the update check: shell:true would
  // interpolate untrusted paths, so keep cmd.exe's one unavoidable parse
  // closed instead.
  const viaCmd: boolean = process.platform === 'win32' && /\.cmd$/i.test(command)

  if (viaCmd && [command, ...args].some((value: string): boolean => /["%&|<>\r\n]/.test(value))) {
    const message = 'The pre-flight snapshot contains an unsafe Windows command argument.'

    log(`[updates] state.db pre-flight failed: ${message}`)
    throw new Error(message)
  }

  const child = spawn(
    viaCmd ? (process.env.ComSpec ?? 'cmd.exe') : command,
    viaCmd
      ? ['/d', '/v:off', '/s', '/c', `""${command}" ${args.map((arg: string): string => `"${arg}"`).join(' ')}"`]
      : args,
    hiddenWindowsChildOptions({ stdio: ['ignore', 'pipe', 'pipe'], windowsVerbatimArguments: viaCmd })
  )

  let stdout = ''
  let stderr = ''

  // Heartbeats reset the no-progress window; they are progress telemetry, not
  // error detail, so they are filtered out of the failure message.
  let timer: NodeJS.Timeout | undefined = undefined
  let progressed = false

  const arm = (ms: number): void => {
    if (timer !== undefined) {
      clearTimeout(timer)
    }

    timer = setTimeout((): void => {
      terminate(child)
    }, ms)
    timer.unref?.()
  }

  const track = (chunk: Buffer): void => {
    const text: string = chunk.toString('utf8')

    if (text.includes(PROGRESS_PREFIX)) {
      progressed = true
      arm(stallMs)
    }

    stderr += text
  }

  child.stdout?.on('data', (chunk: Buffer): void => {
    stdout += chunk
  })
  child.stderr?.on('data', track)

  const settled: Promise<number | null> = new Promise((resolve, reject): void => {
    child.once('error', reject)
    child.once('close', (code: number | null): void => resolve(code))
  })

  // A late `error` after `close` (or vice versa) must not become an
  // unhandled rejection for the losing side of the race.
  settled.catch((): void => {})

  arm(timeoutMs)

  try {
    const outcome: number | null = await settled

    if (timer !== undefined) {
      clearTimeout(timer)
    }

    if (child.exitCode !== null && child.exitCode !== 0) {
      const detail: string = stderr
        .split('\n')
        .filter((line: string): boolean => line !== '' && !line.trimStart().startsWith(PROGRESS_PREFIX.trim()))
        .join(' ')
        .trim()

      throw cancelled(
        `state.db pre-flight failed: ${detail || `exit code ${child.exitCode}`}`,
        log
      )
    }

    if (outcome !== 0) {
      const message: string =
        progressed
          ? `state.db pre-flight made no progress for ${Math.round(stallMs / 1000)} s and was cancelled`
          : `state.db pre-flight timed out after ${Math.round(timeoutMs / 1000)} s and was cancelled`

      throw cancelled(message, log)
    }

    log(`[updates] state.db pre-flight: ${stdout.trim()}`)
  } catch (error: unknown) {
    if (error instanceof Error && error.message.startsWith('state.db pre-flight')) {
      throw error
    }

    const message: string =
      `state.db pre-flight failed: ${error instanceof Error ? error.message : String(error)}. ` +
      'Update cancelled before backend shutdown. Update the selected installation with its hermes update command, then retry.'

    log(`[updates] ${message}`)
    throw new Error(message, { cause: error })
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer)
    }
  }
}
