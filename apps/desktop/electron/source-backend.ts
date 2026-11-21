import { existsSync } from 'node:fs'
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
  /** Per-attempt probe budget (ms). Defaults to PROBE_TIMEOUT_MS. */
  probeTimeoutMs?: number
  /**
   * What a probe that times out on every attempt means.
   *
   * A timeout is "unknown", not "broken": the launcher answered slowly, it did not
   * say the install is gone. The two callers need opposite answers, so the policy is
   * explicit rather than a widened default.
   *
   *  - 'reject' (default) — an unproven runtime is not runnable. Keeps
   *    `isSourceRuntimeUsable()` strict so `chooseUpdaterArgs()` picks `--repair`
   *    for unproven pairs (`windows-hermes-path.ts`: only a runnable pair may
   *    use `--update`).
   *  - 'accept' — the boot rung, where the only alternative is the first-run setup
   *    card. Presence beats a probe that never ran, so a stalled machine must not be
   *    reported as a missing install (#79087).
   */
  onProbeTimeout?: 'reject' | 'accept'
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

  const backend: SourceBackend = {
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

  try {
    // execProbe already retries a timeout once internally, so this is the whole
    // budget: two invocations, not a loop that multiplies that retry (#157543 review).
    await execProbe(command, ['--version'], {
      cwd: root,
      env: { ...process.env, ...options.env, ...env },
      shell,
      stdio: 'ignore',
      timeout: options.probeTimeoutMs ?? PROBE_TIMEOUT_MS,
      windowsHide: true
    })

    return backend
  } catch (err) {
    // A definitive failure (ENOENT, non-zero exit) proves the install is gone.
    if (!isTimeoutError(err)) {
      return null
    }

    return (options.onProbeTimeout ?? 'reject') === 'accept' ? backend : null
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
