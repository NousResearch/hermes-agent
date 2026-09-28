import { existsSync, readdirSync } from 'node:fs'
import path from 'node:path'

import { buildDesktopBackendEnv } from './backend-env'
import { execProbe, isTimeoutError, PROBE_TIMEOUT_MS } from './backend-probes'
import { resolveInstallationLauncher } from './updater-process'

export interface SourceBackend {
  kind: 'command' | 'python'
  label: string
  command: string
  args: string[]
  env: NodeJS.ProcessEnv
  root: string
  bootstrap: false
  shell: boolean
  local: 'installed'
}

interface SourceOptions {
  isWindows?: boolean
  env?: NodeJS.ProcessEnv
}

/** Human-readable probe failure for desktop.log; execProbe already classifies timeouts. */
function probeFailureDetail(err: unknown): string {
  const e = err as { code?: string | number; signal?: string | null }

  // String codes are spawn failures (ENOENT, EACCES, ...), not exit codes.
  if (typeof e?.code === 'string') {
    return e.code === 'ENOENT'
      ? 'launcher is missing or not executable (ENOENT)'
      : `could not be launched (${e.code})`
  }

  const parts: string[] = []

  if (typeof e?.code === 'number') {
    parts.push(`exit code ${e.code}`)
  }

  if (e?.signal) {
    parts.push(`signal ${e.signal}`)
  }

  return parts.length > 0 ? parts.join(' / ') : 'unknown failure'
}

/**
 * Did the desktop installer ever populate this root? A `.hermes-bootstrap-complete`
 * marker attests a completed desktop first-run install; a `node_modules` dir with
 * real entries attests at least a staged dependency install. A bare directory (or
 * one never touched by the installer) is the ordinary "not installed yet" state
 * and must stay quiet (#123921 review P2: the unconditional line fired on every
 * first run and flooded the bounded ring).
 */
function rootHasInstallEvidence(root: string): boolean {
  try {
    if (existsSync(path.join(root, '.hermes-bootstrap-complete'))) {
      return true
    }

    const nodeModules: string = path.join(root, 'node_modules')

    return existsSync(nodeModules) && readdirSync(nodeModules).length > 0
  } catch {
    return false
  }
}

/** Keep the validated command. PM owns interpreter and generation selection. */
export async function resolveSourceInstallationBackend(
  root: string,
  args: string[],
  options: SourceOptions & { hermesHome?: string; log?: (message: string) => void } = {}
): Promise<SourceBackend | null> {
  const log: (message: string) => void = options.log ?? (() => {})

  if (!existsSync(path.join(root, 'hermes_cli', 'main.py'))) {
    // "Never installed" is the ordinary first-run outcome here (main.ts treats
    // the null as the recoverable bootstrap-needed state), so it must stay
    // quiet: this resolver runs on every launch, and a defect-worded line per
    // run would flood the bounded rememberLog ring and evict the lines that
    // explain a real failure. A root the desktop installer POPULATED but that
    // lost hermes_cli/main.py is a torn install — that one is worth a line.
    if (rootHasInstallEvidence(root)) {
      log(
        `Active install root ${root} was populated (desktop bootstrap marker or staged dependencies present) ` +
          'but hermes_cli/main.py is missing; the install looks torn.'
      )
    }

    return null
  }

  const isWindows: boolean = options.isWindows ?? process.platform === 'win32'
  const launcher: string | null = resolveInstallationLauncher(root, isWindows, options.hermesHome)

  if (!launcher) {
    log(
      `No Hermes launcher found for install root ${root} (looked under ${path.join(root, '.hermes', 'bin')} ` +
        'and the historical launcher locations); cannot verify the install.'
    )

    return null
  }

  const shell: boolean = isWindows && /\.(cmd|bat)$/i.test(launcher)
  const command: string = shell ? `"${launcher}"` : launcher
  const env: NodeJS.ProcessEnv = buildDesktopBackendEnv({ currentEnv: options.env ?? process.env })

  try {
    await execProbe(command, ['--version'], {
      cwd: root,
      env: { ...process.env, ...options.env, ...env },
      shell,
      stdio: 'ignore',
      timeout: PROBE_TIMEOUT_MS,
      windowsHide: true
    })
  } catch (err) {
    const reason: string = isTimeoutError(err)
      ? `timed out: each of the 2 attempts (one cold-start retry included) got ${PROBE_TIMEOUT_MS}ms, up to ${PROBE_TIMEOUT_MS * 2}ms total`
      : `failed (${probeFailureDetail(err)})`

    log(`${launcher} --version probe ${reason}; treating the install at ${root} as unusable.`)

    return null
  }

  return {
    kind: 'command',
    label: `Hermes at ${root}`,
    command,
    args: [...args],
    env,
    root,
    bootstrap: false,
    shell,
    local: 'installed'
  }
}

/** Developer overrides retain their interpreter, even outside the checkout. */
export function createSourcePythonBackend(
  root: string,
  python: string | null,
  args: string[],
  options: SourceOptions = {}
): SourceBackend | null {
  if (!python) {
    return null
  }

  let command: string = python

  if ((options.isWindows ?? process.platform === 'win32') && /[\\/]pythonw\.exe$/i.test(python)) {
    // Use only the console interpreter beside the selected windowless one.
    const consolePython: string = python.replace(/pythonw\.exe$/i, 'python.exe')

    if (existsSync(consolePython)) {
      command = consolePython
    }
  }

  return {
    kind: 'python',
    label: `Hermes source at ${root}`,
    command,
    args: ['-m', 'hermes_cli.main', ...args],
    // The backend runs in the user's workspace cwd, and the selected
    // interpreter need not have this checkout installed: name it explicitly.
    // (The scrubbed inherited value could point at another checkout.)
    env: { ...buildDesktopBackendEnv({ currentEnv: options.env ?? process.env }), PYTHONPATH: root },
    root,
    bootstrap: false,
    shell: false,
    local: 'installed'
  }
}
