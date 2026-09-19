/** Windows uninstall delegates fleet draining to the CLI before deleting files. */

import { execFileSync, type ExecFileSyncOptionsWithStringEncoding } from 'node:child_process'
import fs from 'node:fs'

export interface StopGatewayBeforeUpdateDeps {
  /** Defaults to process.platform === 'win32'; injectable for tests. */
  isWindows?: boolean
  /** Defaults to fs.existsSync; injectable for tests. */
  existsSync?: (p: string) => boolean
  /** Defaults to execFileSync from node:child_process; injectable for tests. */
  execFileSync?: (command: string, args: string[], options: ExecFileSyncOptionsWithStringEncoding) => Buffer | string
  /** Observability hook for tests. */
  spy?: (command: string, args: string[]) => void
}

export const GATEWAY_STOP_TIMEOUT_MS = 20_000

/** Best-effort all-profile drain for uninstall. The deletion lock gate follows. */
export function stopGatewayBeforeUpdate(
  hermesCliPath: string,
  hermesHome: string,
  deps: StopGatewayBeforeUpdateDeps = {}
): boolean {
  return runGatewayLifecycleCommand(hermesCliPath, ['gateway', 'stop', '--all'], deps)
}


function runGatewayLifecycleCommand(hermesCliPath: string, args: string[], deps: StopGatewayBeforeUpdateDeps): boolean {
  const isWindows = deps.isWindows ?? process.platform === 'win32'

  if (!isWindows) {
    return false
  }

  const existsSync = deps.existsSync ?? fs.existsSync
  const exec = deps.execFileSync ?? execFileSync

  if (deps.spy) {
    deps.spy(hermesCliPath, args)
  }

  if (!existsSync(hermesCliPath)) {
    return false
  }

  try {
    exec(hermesCliPath, args, {
      timeout: GATEWAY_STOP_TIMEOUT_MS,
      windowsHide: true,
      stdio: 'ignore',
      encoding: 'utf8'
    })

    return true
  } catch {
    // Best-effort (see header comment).
    return false
  }
}
