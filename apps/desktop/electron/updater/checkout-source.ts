import { execFile } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { promisify } from 'node:util'

import { buildDesktopBackendEnv } from '../backend-env'
import { resolveInstallationLauncher } from '../updater-process'
import { hiddenWindowsChildOptions } from '../windows-child-options'

import type { UpdaterStatusWire } from './index'

export interface SourceUpdate extends UpdaterStatusWire {}

export interface SourceUpdateProbe {
  python: string | null
  git: string
  updateRoot: string
  hermesHome: string
  branch?: string
  channel?: 'main' | 'stable' | 'canary'
  force?: boolean
  cachePath?: string
  branchConfigPath?: string
}

const execute: typeof execFile.__promisify__ = promisify(execFile)

export function sourceUpdateEnvironment(updateRoot: string, hermesHome: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    ...buildDesktopBackendEnv(),
    HERMES_HOME: hermesHome,
    HERMES_INSTALL_ROOT: updateRoot
  }

  delete env.HERMES_RUNTIME_DIR

  return env
}

/** Python owns config identity and publication checks; this bridge only transports them. */
export async function readSourceUpdate(probe: SourceUpdateProbe): Promise<SourceUpdate | null> {
  // The install launcher boots PM's committed Python and dependency generation.
  // A PATH Python can import this checkout yet lack its selected dependencies.
  const managed = existsSync(path.join(probe.updateRoot, 'pm'))

  const launcher = managed
    ? resolveInstallationLauncher(probe.updateRoot, process.platform === 'win32', probe.hermesHome)
    : null

  if (managed && !launcher) {
    throw new Error('The source installation launcher is missing; repair this installation before checking updates.')
  }

  if (!managed && !probe.python) {
    throw new Error('No Python interpreter is available to check the source update channel.')
  }

  const args: string[] = [
    ...(managed
      ? ['--run-module', 'hermes_cli.source_check']
      : [
          '-c',
          // Inspect the target checkout's callable, not stderr strings or an editable
          // install elsewhere on sys.path. Exceptions inside a present probe propagate.
          'from pathlib import Path; import runpy; p = Path("hermes_cli/source_check.py"); entry = runpy.run_path(str(p)).get("main") if p.is_file() else None; entry() if callable(entry) else print("null")'
        ]),
    '--install-root',
    probe.updateRoot,
    '--home',
    probe.hermesHome,
    '--git',
    probe.git,
    ...(probe.branch ? ['--branch', probe.branch] : []),
    ...(probe.channel ? ['--channel', probe.channel] : []),
    ...(probe.force ? ['--force'] : []),
    ...(probe.cachePath ? ['--cache-path', probe.cachePath] : []),
    ...(probe.branchConfigPath ? ['--branch-config-path', probe.branchConfigPath] : [])
  ]

  const command: string = (managed ? launcher : probe.python)!
  const viaCmd: boolean = process.platform === 'win32' && /\.cmd$/i.test(command)

  // Node refuses direct .cmd execFile; shell:true interpolates untrusted branch
  // and path arguments. Keep cmd.exe's one unavoidable parse fail-closed.
  if (viaCmd && [command, ...args].some((value: string): boolean => /["%&|<>^\r\n]/.test(value))) {
    throw new Error('The source check contains an unsafe Windows command argument.')
  }

  const result: { stdout: string; stderr: string } = await execute(
    viaCmd ? (process.env.ComSpec ?? 'cmd.exe') : command,
    viaCmd
      ? ['/d', '/v:off', '/s', '/c', `""${command}" ${args.map((arg: string): string => `"${arg}"`).join(' ')}"`]
      : args,
    hiddenWindowsChildOptions({
      cwd: probe.updateRoot,
      env: sourceUpdateEnvironment(probe.updateRoot, probe.hermesHome),
      encoding: 'utf8',
      timeout: 360000,
      maxBuffer: 1024 * 1024,
      windowsVerbatimArguments: viaCmd
    })
  )

  const selection: SourceUpdate | null = JSON.parse(result.stdout) as SourceUpdate | null

  if (selection === null) {
    return null
  }

  if (typeof selection.supported !== 'boolean') {
    throw new Error('The source update check returned an invalid status.')
  }

  // The Python banner uses -1 for an available update with no exact count.
  if (selection.behind === -1) {
    selection.behind = null
  }

  return selection
}