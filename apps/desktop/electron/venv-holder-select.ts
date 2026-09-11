/**
 * venv-holder-select.ts
 *
 * Pure venv-holder selection logic for the Windows pre-update hand-off
 * (testable without Electron).
 *
 * The hand-off kills Hermes-OWNED venv processes so the updater never races a
 * mapped shim or `.pyd`. External holders — a user terminal running `hermes`,
 * unrelated scripts — must NOT be killed: scanVenvBlockers reports them and
 * `main.ts` (releaseBackendLock / applyUpdates) aborts the hand-off instead.
 *
 * Exactly one process shape is owned here: the memory plugin's `hindsight_api`
 * daemon. It is spawned DETACHED, so it outlives the backend tree-kill and
 * keeps `venv\Lib\site-packages\*.pyd` mapped. Selection is an ordinal path
 * prefix plus a cmdline match — no PowerShell `-like` wildcards, whose
 * metacharacters in an install path are a correctness hazard.
 */

/** Ordinal case-insensitive prefix check for Windows paths. */
export function hasWindowsPathPrefix(exePath: string, venvScriptsDir: string): boolean {
  const prefix = `${venvScriptsDir}\\`

  return exePath.length >= prefix.length && exePath.slice(0, prefix.length).toLowerCase() === prefix.toLowerCase()
}

/**
 * True when a process is a Hermes-owned venv daemon: its exe lives under
 * `<venv>\Scripts\` (ordinal case-insensitive prefix) AND its cmdline
 * references `hindsight_api.main`.
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
