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
   * Wall-clock cap for the snapshot subprocess. The previous synchronous
   * `execFileSync(..., { timeout: 30_000 })` killed legitimately slow copies:
   * `backup_sqlite.py` bounds only how long the source stays *locked*
   * (`_safe_copy_db`'s 10 s busy deadline), not how long copying a large
   * state.db takes, so healthy installs died at 30 s with
   * `spawnSync ... ETIMEDOUT` and the update cancelled before it started
   * (#124972).
   */
  timeoutMs?: number
}

/** SIGTERM first, then SIGKILL after a grace period, so the Python side can unlink its staging file. */
function terminate(child: ReturnType<typeof spawn>): void {
  child.kill('SIGTERM')

  setTimeout((): void => {
    if (child.exitCode === null && child.signalCode === null) {
      child.kill('SIGKILL')
    }
  }, 2_000).unref()
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
 */
export async function preflightStateDb({
  python,
  script,
  home,
  log,
  launcher = null,
  timeoutMs = 180_000
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

  child.stdout?.on('data', (chunk: Buffer): void => {
    stdout += chunk
  })
  child.stderr?.on('data', (chunk: Buffer): void => {
    stderr += chunk
  })

  const settled: Promise<number | null> = new Promise((resolve, reject): void => {
    child.once('error', reject)
    child.once('close', (code: number | null): void => resolve(code))
  })

  // A late `error` after `close` (or vice versa) must not become an
  // unhandled rejection for the losing side of the race.
  settled.catch((): void => {})

  let timer: NodeJS.Timeout | undefined = undefined

  try {
    const outcome: number | null | 'expired' = await Promise.race([
      settled,
      new Promise((resolve): void => {
        timer = setTimeout((): void => {
          terminate(child)
          resolve('expired')
        }, timeoutMs)
      })
    ])

    if (outcome === 'expired') {
      const message: string =
        `state.db pre-flight timed out after ${Math.round(timeoutMs / 1000)} s and was cancelled. ` +
        'The snapshot could not complete in time (a large or busy state.db — this is not evidence of corruption). ' +
        'Update cancelled before backend shutdown. Update the selected installation with its hermes update command, then retry.'

      log(`[updates] ${message}`)
      throw new Error(message)
    }

    if (outcome !== 0) {
      const detail: string = stderr.trim() || `exit code ${outcome}`
      const message: string =
        `state.db pre-flight failed: ${detail}. ` +
        'Update cancelled before backend shutdown. Update the selected installation with its hermes update command, then retry.'

      log(`[updates] ${message}`)
      throw new Error(message)
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
