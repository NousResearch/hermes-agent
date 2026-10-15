import { execFile } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { promisify } from 'node:util'

import { hiddenWindowsChildOptions } from '../windows-child-options'

const DEFAULT_TIMEOUT_MS = 5 * 60_000
const MAX_TIMEOUT_MS = 30 * 60_000
const execFileAsync = promisify(execFile)
const COPY_RATE_BYTES_PER_SECOND = 5 * 1024 * 1024
const TIMEOUT_SLACK_MS = 60_000

export function stateDbPreflightTimeoutMs(home: string): number {
  let bytes = 0

  for (const name of ['state.db', 'state.db-wal']) {
    try {
      bytes += fs.statSync(path.join(home, name)).size
    } catch {
      // Missing WAL is normal; an unreadable size retains the conservative floor.
    }
  }

  return Math.min(MAX_TIMEOUT_MS, Math.max(
    DEFAULT_TIMEOUT_MS,
    Math.ceil(bytes / COPY_RATE_BYTES_PER_SECOND) * 1_000 + TIMEOUT_SLACK_MS
  ))
}

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
}

// Await completion before stopping the backend, while keeping Electron responsive.
export async function preflightStateDb({ python, script, home, log, launcher = null }: StateDbPreflight): Promise<void> {
  try {
    const command: string | null = launcher ?? python

    if (!command) {
      throw new Error('Python not found')
    }

    const args: string[] = launcher ? ['--run-module', 'hermes_cli.backup_sqlite', home] : ['-I', '-S', script, home]

    // Node refuses direct .cmd execFile; an older published launcher can still
    // be one. Same fail-closed guard as the update check: shell:true would
    // interpolate untrusted paths, so keep cmd.exe's one unavoidable parse
    // closed instead.
    const viaCmd: boolean = process.platform === 'win32' && /\.cmd$/i.test(command)

    if (viaCmd && [command, ...args].some((value: string): boolean => /["%&|<>^\r\n]/.test(value))) {
      throw new Error('The pre-flight snapshot contains an unsafe Windows command argument.')
    }

    const { stdout } = await execFileAsync(
      viaCmd ? (process.env.ComSpec ?? 'cmd.exe') : command,
      viaCmd
        ? ['/d', '/v:off', '/s', '/c', `""${command}" ${args.map((arg: string): string => `"${arg}"`).join(' ')}"`]
        : args,
      hiddenWindowsChildOptions({
        encoding: 'utf8',
        timeout: stateDbPreflightTimeoutMs(home),
        windowsVerbatimArguments: viaCmd
      })
    )

    log(`[updates] state.db pre-flight: ${stdout.trim()}`)
  } catch (error: unknown) {
    const message =
      `state.db pre-flight failed: ${error instanceof Error ? error.message : String(error)}. ` +
      'Update cancelled before backend shutdown. Update the selected installation with its hermes update command, then retry.'

    log(`[updates] ${message}`)
    throw new Error(message, { cause: error })
  }
}
