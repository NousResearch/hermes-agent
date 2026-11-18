import { existsSync } from 'node:fs'
import path from 'node:path'

import { buildDesktopBackendEnv } from './backend-env'
import { execProbe, isTimeoutError, PROBE_TIMEOUT_MS } from './backend-probes'
import { resolveInstallationLauncher } from './updater-process'

/**
 * Attempts for the usability probe. A stalled machine must not be mistaken for a
 * missing install: a timeout means "unknown", and only a definitive failure
 * (ENOENT, non-zero exit) proves the install is gone.
 */
const PROBE_ATTEMPTS = 3
const PROBE_RETRY_BACKOFF_MS = 1_000

const delay = (ms: number): Promise<void> =>
  new Promise((resolve): void => {
    setTimeout((): void => {
      resolve()
    }, ms)
  })

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
  /** Per-attempt probe budget (ms). Defaults to PROBE_TIMEOUT_MS. */
  probeTimeoutMs?: number
  /** How many times a timed-out probe is re-attempted. Defaults to PROBE_ATTEMPTS. */
  probeAttempts?: number
  /** Delay between timed-out attempts (ms). Defaults to PROBE_RETRY_BACKOFF_MS. */
  probeBackoffMs?: number
}

/** Keep the validated command. PM owns interpreter and generation selection. */
export async function resolveSourceInstallationBackend(
  root: string,
  args: string[],
  options: SourceOptions & { hermesHome?: string } = {}
): Promise<SourceBackend | null> {
  if (!existsSync(path.join(root, 'hermes_cli', 'main.py'))) {
    return null
  }

  const isWindows: boolean = options.isWindows ?? process.platform === 'win32'
  const launcher: string | null = resolveInstallationLauncher(root, isWindows, options.hermesHome)

  if (!launcher) {
    return null
  }

  const shell: boolean = isWindows && /\.(cmd|bat)$/i.test(launcher)
  const command: string = shell ? `"${launcher}"` : launcher
  const env: NodeJS.ProcessEnv = buildDesktopBackendEnv({ currentEnv: options.env ?? process.env })

  const timeoutMs: number = options.probeTimeoutMs ?? PROBE_TIMEOUT_MS
  const attempts: number = Math.max(1, options.probeAttempts ?? PROBE_ATTEMPTS)
  const backoffMs: number = options.probeBackoffMs ?? PROBE_RETRY_BACKOFF_MS

  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      await execProbe(command, ['--version'], {
        cwd: root,
        env: { ...process.env, ...options.env, ...env },
        shell,
        stdio: 'ignore',
        timeout: timeoutMs,
        windowsHide: true
      })

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
    } catch (err) {
      // A timeout means "unknown", not "broken": a stalled machine must never be
      // reported as a missing install. Only a definitive failure (ENOENT,
      // non-zero exit) fails fast.
      if (!isTimeoutError(err)) {
        return null
      }

      if (attempt < attempts - 1) {
        await delay(backoffMs)
      }
    }
  }

  return null
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
