import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'

import { resolveInstallationLauncher } from '../updater-process'
import { hiddenWindowsChildOptions } from '../windows-child-options'

import { sourceUpdateEnvironment } from './checkout-source'

/** The executable recipe for one emergency-snapshot invocation. */
export interface StateDbSnapshotRunner {
  command: string
  args: string[]
  /** Windows .cmd launcher: cmd.exe re-parses the line, so argv stays verbatim. */
  viaCmd: boolean
  cwd?: string
  env?: NodeJS.ProcessEnv
}

interface StateDbSnapshotInput {
  python: string | null
  updateRoot: string
  script: string
  home: string
  isWindows?: boolean
}

interface StateDbPreflight extends StateDbSnapshotInput {
  log: (message: string) => void
}

// cmd.exe re-parses its command line (CVE-2024-27980 workaround — same screen
// as updater/checkout-source.ts): any of these in a path or argument would
// change the command instead of naming a file.
const CMD_UNSAFE_ARG: RegExp = /["%&|<>\r\n]/

/**
 * The interpreter that runs the emergency snapshot. The checkout's own Python
 * (including the HERMES_DESKTOP_PYTHON override) stays the first rung: the
 * snapshot script is stdlib-only, so any resolved interpreter is the faster,
 * dependency-free invocation. PM deletes the in-tree venv/.venv once a
 * generation is committed (source-python.ts), so null is the ordinary answer
 * on a managed install — there the installation launcher boots PM's committed
 * generation, the same runtime readSourceUpdate's managed rung already uses
 * for the update probe (#122991).
 */
export function resolveStateDbSnapshotRunner({
  python,
  updateRoot,
  script,
  home,
  isWindows = process.platform === 'win32'
}: StateDbSnapshotInput): StateDbSnapshotRunner | null {
  if (python) {
    return { command: python, args: ['-I', '-S', script, home], viaCmd: false }
  }

  if (!existsSync(path.join(updateRoot, 'pm'))) {
    return null
  }

  const launcher: string | null = resolveInstallationLauncher(updateRoot, isWindows, home)

  if (!launcher) {
    return null
  }

  const args: string[] = ['--run-module', 'hermes_cli.backup_sqlite', home]
  const viaCmd: boolean = isWindows && /\.(cmd|bat)$/i.test(launcher)

  if (viaCmd && [launcher, ...args].some((value: string): boolean => CMD_UNSAFE_ARG.test(value))) {
    throw new Error('The state.db pre-flight contains an unsafe Windows command argument.')
  }

  return {
    command: launcher,
    args,
    viaCmd,
    cwd: updateRoot,
    env: sourceUpdateEnvironment(updateRoot, home)
  }
}

// Synchronous by design: the caller must not stop the backend before the snapshot.
export function preflightStateDb({ python, updateRoot, script, home, log, isWindows }: StateDbPreflight): void {
  try {
    const runner: StateDbSnapshotRunner | null = resolveStateDbSnapshotRunner({ python, updateRoot, script, home, isWindows })

    if (!runner) {
      throw new Error('Python not found (no checkout interpreter and no installation launcher)')
    }

    const viaCmd: boolean = runner.viaCmd
    const command: string = viaCmd ? (process.env.ComSpec ?? 'cmd.exe') : runner.command

    const args: string[] = viaCmd
      ? ['/d', '/s', '/c', `""${runner.command}" ${runner.args.map((arg: string): string => `"${arg}"`).join(' ')}"`]
      : runner.args

    const result: string = execFileSync(
      command,
      args,
      hiddenWindowsChildOptions({
        encoding: 'utf8',
        timeout: 30_000,
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsVerbatimArguments: viaCmd,
        ...(runner.cwd ? { cwd: runner.cwd } : {}),
        ...(runner.env ? { env: runner.env } : {})
      })
    )

    log(`[updates] state.db pre-flight: ${result.trim()}`)
  } catch (error: unknown) {
    const message: string =
      `state.db pre-flight failed: ${error instanceof Error ? error.message : String(error)}. Update cancelled before backend shutdown. Update the selected installation with its hermes update command, then retry.`

    log(`[updates] ${message}`)
    throw new Error(message, { cause: error })
  }
}
