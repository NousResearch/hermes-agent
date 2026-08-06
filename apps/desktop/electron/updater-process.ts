import { spawn, type SpawnOptions } from 'node:child_process'
import { statSync } from 'node:fs'
import path from 'node:path'

import { hiddenWindowsChildOptions } from './windows-child-options'

export interface UpdaterChild {
  pid?: number
  unref: () => void
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
