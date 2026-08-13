import { spawn, type SpawnOptions } from 'node:child_process'
import { statSync } from 'node:fs'
import path from 'node:path'

import { hiddenWindowsChildOptions } from './windows-child-options'

export interface UpdaterChild {
  pid?: number
  unref: () => void
}

export interface ResolveUpdateScriptHandoffDeps {
  isWindows?: boolean
  fileExists?: (candidate: string) => boolean
}

export interface UpdateScriptHandoff {
  command: string
  args: string[]
  scriptPath: string
}

/**
 * Repo-owned Windows update hand-off (frozen-binary escape hatch).
 *
 * The staged Tauri `hermes-setup.exe` has no self-update path, so every
 * updater-side fix only reaches users when a new binary is built, signed and
 * published — which historically lags main by months and strands users on
 * long-fixed bugs (cache resolver #67369, marker self-adopt #74782; the
 * 2026-08-09 incident chain). `scripts/desktop-update/windows.ps1` lives in the repo
 * checkout instead: every `hermes update` refreshes the code that drives the
 * NEXT update, and only PowerShell itself is frozen.
 *
 * Returns the spawn recipe when the script exists in the checkout, or null
 * (caller falls back to the staged binary — old checkouts that predate the
 * script keep working unchanged). Windows-only by the same policy as
 * resolveStagedUpdaterBinary: POSIX updates in place via
 * applyUpdatesPosixInApp and needs no hand-off at all.
 */
export function resolveUpdateScriptHandoff(
  updateRoot: string,
  deps: ResolveUpdateScriptHandoffDeps = {}
): UpdateScriptHandoff | null {
  const isWindows = deps.isWindows ?? process.platform === 'win32'

  if (!isWindows) {
    return null
  }

  const exists = deps.fileExists ?? stagedFileExists

  // Current layout first, then the pre-reorg flat path — an updated asar can
  // meet a checkout from either side of the move (the checkout also ships a
  // forwarder at the legacy path for the inverse skew).
  for (const candidate of [
    path.join(updateRoot, 'scripts', 'desktop-update', 'windows.ps1'),
    path.join(updateRoot, 'scripts', 'desktop-update.ps1')
  ]) {
    if (exists(candidate)) {
      return {
        command: 'powershell',
        args: ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', candidate],
        scriptPath: candidate
      }
    }
  }

  return null
}

/**
 * Repo-owned POSIX update hand-off (the mac/linux twin of the above).
 *
 * Replaces the in-app posix updater: the Desktop spawns the script detached
 * and QUITS, the script waits it out, runs `hermes update`, swaps/relaunches
 * the app, and writes .hermes-update-result.json. With the app gone before
 * the update starts, the HERMES_DESKTOP_CHILD_PID reaper-exclusion dance is
 * unnecessary — there are no live desktop backends to spare.
 *
 * Null when the checkout predates the script (caller surfaces the manual
 * `hermes update` card — old checkouts pull the script on their next update).
 */
export function resolvePosixScriptHandoff(
  updateRoot: string,
  deps: ResolveUpdateScriptHandoffDeps = {}
): UpdateScriptHandoff | null {
  const isWindows = deps.isWindows ?? process.platform === 'win32'

  if (isWindows) {
    return null
  }

  const scriptPath = path.join(updateRoot, 'scripts', 'desktop-update', 'posix.sh')
  const exists = deps.fileExists ?? stagedFileExists

  if (!exists(scriptPath)) {
    return null
  }

  return {
    command: '/bin/bash',
    args: [scriptPath],
    scriptPath
  }
}

/**
 * Wrap a PowerShell hand-off invocation so it survives a detached, hidden
 * spawn from Electron.
 *
 * Verified empirically (2026-08-09, Windows 11): `spawn('powershell', [...,
 * '-File', script], { detached: true, stdio: 'ignore', windowsHide: true })`
 * exits 0 WITHOUT executing a single line of the script. powershell.exe is a
 * console-subsystem binary; detached+windowsHide gives it no console to
 * attach to, and Windows PowerShell 5.1 dies during console init before
 * -File processing (the same class of failure as #54220's conhost work, on
 * the launch side). The same spawn with a visible console, or non-detached,
 * runs fine — so unit tests and foreground use hide the bug.
 *
 * `cmd /c start "" /min powershell ...` was the variant that survived the
 * full detached+hidden production shape in testing: `start` allocates the
 * child its own (minimized) console and fully detaches it from cmd.exe,
 * which exits immediately. The spawned pid is therefore the WRAPPER's —
 * callers must not use it as a marker owner (the script claims the marker
 * itself with its own $PID).
 */
export function wrapHandoffForDetachedConsole(
  handoff: UpdateScriptHandoff,
  extraArgs: string[]
): {
  command: string
  args: string[]
} {
  return {
    command: 'cmd.exe',
    args: ['/d', '/s', '/c', 'start', '', '/min', handoff.command, ...handoff.args, ...extraArgs]
  }
}

/**
 * Electron/Chromium internal switches that must NOT be replayed on re-exec:
 * runtime artifacts of THIS launch, not user intent (ported from the deleted
 * update-relaunch.ts; #45205). `--no-sandbox` is deliberately kept — it is
 * the user's sandbox opt-out and the signal that makes a relaunch safe when
 * chrome-sandbox isn't setuid.
 */
export const INTERNAL_ARG_PREFIXES = [
  '--type=',
  '--user-data-dir=',
  '--enable-features=',
  '--disable-features=',
  '--field-trial-handle=',
  '--enable-logging',
  '--log-file=',
  '--disable-gpu-sandbox',
  '--lang=',
  '--inspect',
  '--remote-debugging-port='
]

/** Filter Electron internals from process.argv.slice(1) so the relaunched
 * app replays only user/launcher intent (deep links, app flags). */
export function collectRelaunchArgs(argv: unknown): string[] {
  if (!Array.isArray(argv)) {
    return []
  }

  return argv.filter((arg): arg is string => {
    if (typeof arg !== 'string' || arg.length === 0) {
      return false
    }

    return !INTERNAL_ARG_PREFIXES.some(prefix =>
      prefix.endsWith('=') ? arg.startsWith(prefix) : arg === prefix || arg.startsWith(prefix + '=')
    )
  })
}

/** True when the user has opted out of the SUID sandbox — the relaunch is
 * safe even if chrome-sandbox fails preflight (ported from update-relaunch.ts). */
export function sandboxFallbackFromEnv(env: Record<string, string | undefined>, launchArgs: string[]): boolean {
  const disable = String(env?.ELECTRON_DISABLE_SANDBOX || '').trim()

  if (disable === '1' || disable.toLowerCase() === 'true') {
    return true
  }

  return Array.isArray(launchArgs) && launchArgs.includes('--no-sandbox')
}

export interface ResolveStagedUpdaterBinaryDeps {
  isWindows?: boolean
  fileExists?: (candidate: string) => boolean
  stagedMtimeMs?: (candidate: string) => number | null
}

/**
 * Staged installers older than this have no self-PID exclusion in
 * `UpdateMarkerGuard::acquire` and will refuse an update whose marker was
 * pre-written on their behalf.
 *
 * The self-adopt fix landed in #74782 / 160586ff8 (2026-07-30 17:57 +0700).
 * We compare against the start of 2026-07-31 UTC so the boundary is
 * unambiguous for binaries staged that same day.
 */
export const MARKER_SELF_ADOPT_EPOCH_MS = Date.UTC(2026, 6, 31)

function stagedFileExists(candidate: string): boolean {
  try {
    return statSync(candidate).isFile()
  } catch {
    return false
  }
}

function stagedFileMtimeMs(candidate: string): number | null {
  try {
    return statSync(candidate).mtimeMs
  } catch {
    return null
  }
}

/**
 * Decide which staged installer binary — if any — may be handed an update.
 *
 * The Tauri installer self-copies into HERMES_HOME on *every* platform
 * (`hermes-setup.exe` on Windows, `hermes-setup` elsewhere — see
 * apps/bootstrap-installer `paths::installer_dest` and
 * `bootstrap::copy_self_to_hermes_home`), so finding that binary on macOS or
 * Linux is expected, not leftover junk.
 *
 * Handing an update to it is nonetheless a Windows-only policy. Windows needs
 * the quit -> hand-off -> rebuild dance because a venv shim file lock keeps the
 * running desktop from rewriting its own bits; macOS and Linux have no such
 * lock and update in place through applyUpdatesPosixInApp(). Off Windows the
 * hand-off therefore buys nothing and costs a great deal: a staged binary older
 * than the hand-off protocol holds the update marker, spawns `hermes update`,
 * and that child refuses its own parent — wedging the in-app Update button for
 * good, with no route (update, re-download, reinstall) to a newer binary
 * (#74836). Returning null off Windows is what routes those platforms to the
 * in-app updater.
 *
 * Null on Windows too when nothing is staged (a dev/source run, or a CLI
 * install that never went through the installer); callers degrade gracefully.
 */
export function resolveStagedUpdaterBinary(
  hermesHome: string,
  deps: ResolveStagedUpdaterBinaryDeps = {}
): string | null {
  const isWindows = deps.isWindows ?? process.platform === 'win32'

  if (!isWindows) {
    return null
  }

  const fileExists = deps.fileExists ?? stagedFileExists
  const candidate = path.join(hermesHome, 'hermes-setup.exe')

  return fileExists(candidate) ? candidate : null
}

/**
 * True when the staged installer is new enough to survive a pre-written marker.
 *
 * `copy_self_to_hermes_home` deliberately no-ops during `--update`
 * (apps/bootstrap-installer/src-tauri/src/paths.rs), so the binary staged by a
 * user's ORIGINAL install orchestrates every later update — forever. Installers
 * predating #74782 have no self-PID exclusion in `UpdateMarkerGuard::acquire`,
 * so when the desktop pre-writes the marker naming that very updater, the
 * updater reads its own claim as a foreign live owner and aborts with
 * "Another Hermes update is already running (PID <itself>, started 1s ago)" —
 * the observed infinite "Install didn't finish" loop. Skipping the pre-write
 * for those binaries lets them acquire cleanly and run `hermes update`, which
 * pulls the permanent fixes. See shouldPrewriteUpdateMarker.
 *
 * We cannot ask the binary its version without executing it, so use its mtime:
 * the installer is written to HERMES_HOME at install/repair time, making mtime
 * a faithful stamp of which installer generation produced it.
 *
 * Unreadable mtime counts as UNSUPPORTED — the pre-write is a best-effort
 * hardening, while a wedged updater is unrecoverable, so we bias toward the
 * path that can always make progress.
 */
export function stagedUpdaterSupportsPrewrittenMarker(
  candidate: string,
  deps: ResolveStagedUpdaterBinaryDeps = {}
): boolean {
  const mtimeMs = (deps.stagedMtimeMs ?? stagedFileMtimeMs)(candidate)

  return typeof mtimeMs === 'number' && Number.isFinite(mtimeMs) && mtimeMs >= MARKER_SELF_ADOPT_EPOCH_MS
}

export interface SpawnUpdaterProcessDeps {
  isWindows?: boolean
  spawnProcess?: (command: string, args: string[], options: SpawnOptions) => UpdaterChild
}

/**
 * Spawn the detached installer used for update and bootstrap-recovery handoffs.
 * The helper owns both hidden-console selection and unref semantics so every
 * updater handoff follows the same behavior and can be tested without Electron.
 */
export function spawnUpdaterProcess(
  updater: string,
  updaterArgs: string[],
  options: SpawnOptions,
  deps: SpawnUpdaterProcessDeps = {}
): UpdaterChild {
  const isWindows = deps.isWindows ?? process.platform === 'win32'
  const spawnOptions = hiddenWindowsChildOptions(options, isWindows) as SpawnOptions

  const child = deps.spawnProcess
    ? deps.spawnProcess(updater, updaterArgs, spawnOptions)
    : spawn(updater, updaterArgs, spawnOptions)

  child.unref()

  return child
}
