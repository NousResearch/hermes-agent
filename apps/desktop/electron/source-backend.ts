import { existsSync } from 'node:fs'
import path from 'node:path'

import { buildDesktopBackendEnv } from './backend-env'
import { execProbe, PROBE_TIMEOUT_MS } from './backend-probes'
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

/**
 * Windows installers may publish both launchers. Keep the executable as the
 * first choice, but let a working command launcher recover from a policy or
 * compatibility block on the executable before Desktop falls back to setup.
 */
export function installationLauncherCandidates(launcher: string, isWindows: boolean): string[] {
  if (!isWindows || !/\.exe$/i.test(launcher)) {
    return [launcher]
  }

  const commandLauncher: string = launcher.replace(/\.exe$/i, '.cmd')

  return existsSync(commandLauncher) ? [launcher, commandLauncher] : [launcher]
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

  const env: NodeJS.ProcessEnv = buildDesktopBackendEnv({ currentEnv: options.env ?? process.env })

  for (const candidate of installationLauncherCandidates(launcher, isWindows)) {
    const shell: boolean = isWindows && /\.(cmd|bat)$/i.test(candidate)
    const command: string = shell ? `"${candidate}"` : candidate

    try {
      await execProbe(command, ['--version'], {
        cwd: root,
        env: { ...process.env, ...options.env, ...env },
        shell,
        stdio: 'ignore',
        timeout: PROBE_TIMEOUT_MS,
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
    } catch {
      // A blocked or incompatible .exe can have a usable .cmd sibling.
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
