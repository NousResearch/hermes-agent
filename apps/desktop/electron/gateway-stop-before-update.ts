/**
 * gateway-stop-before-update.ts
 *
 * Windows-only helper for the update hand-off (#70337): stop every
 * separately-running messaging gateway BEFORE the venv-shim lock poll.
 *
 * Why not just tree-kill gateway.pid's PID:
 *  - gateway.pid records the uv WORKER process, but the venv shim lock is
 *    held by its parent LAUNCHER (venv\Scripts\python.exe). taskkill /T from
 *    the worker PID does not reach parents, so the lock could survive.
 *  - a single gateway.pid read misses multi-profile setups entirely.
 *
 * So we delegate to `hermes gateway stop --all`: the CLI discovers every
 * profile's gateway processes (launcher + worker) via find_gateway_pids,
 * drains in-flight agents (planned-stop marker -> resume_pending), and
 * force-kills survivors — the same logic `hermes update`'s
 * _pause_windows_gateways_for_update relies on.
 *
 * Pure + dependency-injected so the launcher/worker and multi-profile
 * behavior is assertable without booting Electron.
 */

import { execFile, type ExecFileOptionsWithStringEncoding } from 'node:child_process'
import fs from 'node:fs'

export interface StopGatewayBeforeUpdateDeps {
  /** Defaults to process.platform === 'win32'; injectable for tests. */
  isWindows?: boolean
  /** Defaults to fs.existsSync; injectable for tests. */
  existsSync?: (p: string) => boolean
  /** Defaults to async execFile from node:child_process; injectable for tests. */
  execFile?: (
    command: string,
    args: string[],
    options: ExecFileOptionsWithStringEncoding,
    callback: (error: Error | null) => void
  ) => unknown
  /** Observability hook for tests. */
  spy?: (command: string, args: string[]) => void
}

export const GATEWAY_STOP_TIMEOUT_MS = 20_000

/**
 * Best-effort stop of all-profile messaging gateways via the CLI.
 * Never throws: a wedged/absent CLI must not abort the update hand-off
 * (the shim-lock poll + the updater's venv-blocker scan still fail loudly
 * if the venv stays held). Returns true when the CLI ran (or was invoked
 * with the injected spy), false when skipped (non-Windows / missing CLI).
 */
export async function stopGatewayBeforeUpdate(
  hermesCliPath: string,
  hermesHome: string,
  deps: StopGatewayBeforeUpdateDeps = {}
): Promise<boolean> {
  return await runGatewayLifecycleCommand(hermesCliPath, ['gateway', 'stop', '--all'], deps)
}

/**
 * Drain-semantics counterpart (#76057 review): `gateway stop --all` before
 * the lock gate takes gateways down even when the update later ABORTS
 * (venv-blocked by a user terminal, probe failure, updater spawn failure).
 * The updater's own pause machinery resumes what it pauses — the Desktop
 * must mirror that on its abort paths, or a failed update strands every
 * profile's gateway stopped. Best-effort, never throws.
 */
export async function startGatewaysAfterUpdateAbort(
  hermesCliPath: string,
  deps: StopGatewayBeforeUpdateDeps = {}
): Promise<boolean> {
  return await runGatewayLifecycleCommand(hermesCliPath, ['gateway', 'start', '--all'], deps)
}

async function runGatewayLifecycleCommand(
  hermesCliPath: string,
  args: string[],
  deps: StopGatewayBeforeUpdateDeps
): Promise<boolean> {
  const isWindows = deps.isWindows ?? process.platform === 'win32'

  if (!isWindows) {
    return false
  }

  const existsSync = deps.existsSync ?? fs.existsSync
  const run = deps.execFile ?? execFile

  if (deps.spy) {
    deps.spy(hermesCliPath, args)
  }

  if (!existsSync(hermesCliPath)) {
    return false
  }

  try {
    await new Promise<void>((resolve, reject) => {
      run(hermesCliPath, args, {
        timeout: GATEWAY_STOP_TIMEOUT_MS,
        windowsHide: true,
        encoding: 'utf8'
      }, error => {
        if (error) {
          reject(error)

          return
        }

        resolve()
      })
    })

    return true
  } catch {
    // Best-effort (see header comment).
    return false
  }
}
