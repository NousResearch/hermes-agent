/**
 * venv-holder-select.ts
 *
 * Pure Windows venv-holder selection logic (testable without Electron).
 *
 * The pre-update handoff kills Hermes-OWNED venv daemons (the memory plugin's
 * hindsight daemon) so the updater never races a mapped shim. External
 * holders (a user terminal running `hermes`, unrelated scripts) must NOT be
 * killed — current design reports them via scanVenvBlockers and ABORTS the
 * handoff instead (main.ts releaseBackendLock / applyUpdates).
 */

/** Ordinal case-insensitive prefix check for Windows paths. */
export function hasWindowsPathPrefix(exePath: string, venvScriptsDir: string): boolean {
  const prefix = `${venvScriptsDir}\\`

  return exePath.length >= prefix.length && exePath.slice(0, prefix.length).toLowerCase() === prefix.toLowerCase()
}

/**
 * True when a process is a Hermes-owned venv daemon: its exe lives under
 * `<venv>\Scripts\` (ordinal case-insensitive prefix) AND its cmdline
 * references `hindsight_api.main` (the memory daemon the memory plugin
 * spawns DETACHED — it outlives Hermes and holds venv shims mapped).
 */
export function isHermesOwnedVenvDaemon(
  exePath: string | null | undefined,
  cmdline: string | null | undefined,
  venvScriptsDir: string
): boolean {
  if (!exePath || !cmdline) {
    return false
  }

  return hasWindowsPathPrefix(exePath, venvScriptsDir) && /hindsight_api\.main/i.test(cmdline)
}

/**
 * True when a process is an external Hermes process holding this install's venv
 * (#62311): its exe lives under `<venv>\Scripts\` AND it is unambiguously a
 * Hermes program — the `hermes.exe` shim, `python -m hermes_cli...`, or
 * `python -m hermes ...`. These are the autostart holders (the gateway Startup
 * item, the dashboard Scheduled Task) that neither the desktop's backend
 * teardown nor the hindsight-daemon sweep reach, and that keep the venv shim
 * locked so the update hand-off aborts every time.
 *
 * Deliberately NARROWER than a bare path/cmdline substring against the install
 * root (the approach that sank #62445): an unrelated process that merely
 * mentions the install root or borrows the venv interpreter for its own script
 * must NOT be tree-killed. Non-Hermes venv users still abort the hand-off via
 * the shim-lock probe instead.
 */
export function isExternalVenvHolder(
  exePath: string | null | undefined,
  cmdline: string | null | undefined,
  venvScriptsDir: string
): boolean {
  if (!exePath || !cmdline) {
    return false
  }

  if (!hasWindowsPathPrefix(exePath, venvScriptsDir)) {
    return false
  }

  const exeName = exePath.slice(exePath.lastIndexOf('\\') + 1).toLowerCase()

  if (exeName === 'hermes.exe') {
    return true
  }

  return /hermes_cli/i.test(cmdline) || /(^|\s|")-m\s+hermes([.\s"']|$)/i.test(cmdline)
}
