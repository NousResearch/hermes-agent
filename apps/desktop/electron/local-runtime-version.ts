import { execFile } from 'node:child_process'

import { buildDesktopBackendEnv } from './backend-env'
import { PROBE_TIMEOUT_MS } from './backend-probes'
import { resolveInstallationLauncher } from './updater-process'
import { windowsShellCommand } from './windows-child-options'

export type VersionProbeRunner = (
  command: string,
  args: string[],
  options: {
    cwd: string
    env: NodeJS.ProcessEnv
    timeout: number
    maxBuffer: number
    shell: boolean
    windowsHide: boolean
  }
) => Promise<string>

const DEFAULT_CACHE_TTL_MS = 5_000
let cached: { launcher: string; version: string; expiresAt: number } | null = null
let inFlight: { launcher: string; promise: Promise<string> } | null = null

/** Extract the client version printed by the exact local installation launcher. */
export function parseLocalRuntimeVersion(output: string): string {
  const match = output.match(/^\s*Hermes Agent\s+([^\s(]+)/m)
  return match?.[1]?.replace(/^v/, '') ?? ''
}
function runVersionProbe(
  command: string,
  args: string[],
  options: Parameters<VersionProbeRunner>[2]
): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(command, args, { ...options, encoding: 'utf8' }, (error, stdout) => {
      if (error) {
        reject(error)
        return
      }

      resolve(stdout)
    })
  })
}

/**
 * Read the version from the local source installation without contacting a
 * gateway or starting the full backend. The short cache prevents repeated
 * About/status refreshes from spawning the launcher unnecessarily.
 */
export async function resolveLocalRuntimeVersion(
  root: string,
  hermesHome: string,
  options: {
    isWindows?: boolean
    launcher?: string | null
    run?: VersionProbeRunner
    cacheTtlMs?: number
  } = {}
): Promise<string> {
  const isWindows = options.isWindows ?? process.platform === 'win32'
  let launcher: string | null

  try {
    launcher = options.launcher === undefined
      ? resolveInstallationLauncher(root, isWindows, hermesHome)
      : options.launcher
  } catch {
    return ''
  }

  if (!launcher) {
    return ''
  }

  const run = options.run ?? runVersionProbe
  const useCache = options.run === undefined
  const now = Date.now()

  if (useCache && cached?.launcher === launcher && cached.expiresAt > now) {
    return cached.version
  }

  if (useCache && inFlight?.launcher === launcher) {
    return inFlight.promise
  }

  const shell = isWindows && /\.(cmd|bat)$/i.test(launcher)
  const command = windowsShellCommand(launcher, shell, isWindows)
  const promise = run(command, ['--version'], {
    cwd: root,
    env: { ...process.env, ...buildDesktopBackendEnv({ currentEnv: process.env }), HERMES_HOME: hermesHome },
    timeout: PROBE_TIMEOUT_MS,
    maxBuffer: 64 * 1024,
    shell,
    windowsHide: true
  })
    .then(parseLocalRuntimeVersion)
    .catch(() => '')

  if (useCache) {
    inFlight = { launcher, promise }
    void promise.then(version => {
      cached = { launcher, version, expiresAt: Date.now() + (options.cacheTtlMs ?? DEFAULT_CACHE_TTL_MS) }
    }).finally(() => {
      if (inFlight?.promise === promise) {
        inFlight = null
      }
    })
  }

  return promise
}
