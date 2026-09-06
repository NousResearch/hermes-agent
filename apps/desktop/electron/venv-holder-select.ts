/**
 * venv-holder-select.ts
 *
 * Pure Windows venv-holder selection logic (testable without Electron).
 *
 * The pre-update handoff kills Hermes-OWNED venv processes so the updater
 * never races a mapped shim or .pyd. External holders (a user terminal
 * running `hermes`, unrelated scripts) must NOT be killed — current design
 * reports them via scanVenvBlockers and ABORTS the handoff instead
 * (main.ts releaseBackendLock / applyUpdates).
 *
 * On this host the processes that actually keep venv\Lib\site-packages\*.pyd
 * mapped are not the hindsight daemon. They are:
 *   - the operator-managed Tailscale `hermes serve` (serve.pid / port 9119)
 *     and its execute_code kernel children
 *   - Codex/Claude `hermes_tools_mcp_server` bridges launched from this venv
 *   - the matching .hermes-runtime interpreter for each venv\Scripts trampoline
 *
 * Upstream's preferred path (#101502 / kernel-proven holders) names those
 * processes and then fail-closes inside a 20s SuperF4-per-PID budget. This
 * selector is the host-specific drain that runs *before* that budget: identity
 * by exe prefix + argv shape, then taskkill /T of those PIDs only.
 */

/** Ordinal case-insensitive prefix check for Windows paths. */
export function hasWindowsPathPrefix(exePath: string, venvScriptsDir: string): boolean {
  const prefix = `${venvScriptsDir}\\`

  return exePath.length >= prefix.length && exePath.slice(0, prefix.length).toLowerCase() === prefix.toLowerCase()
}

/** WMI filter: do not table-scan every process on a loaded workstation. */
export function venvHolderProcessWmiFilter(): string {
  return "Name='python.exe' OR Name='pythonw.exe'"
}

/** PowerShell listing used by the pre-update daemon reap. */
export function buildVenvHolderListCommand(): string {
  return (
    `Get-CimInstance Win32_Process -Filter "${venvHolderProcessWmiFilter()}" | ` +
    'Where-Object { $_.ExecutablePath -and $_.CommandLine } | ' +
    'Select-Object ProcessId, ExecutablePath, CommandLine | ConvertTo-Json -Compress'
  )
}

export function parseServePidFile(raw: string | null | undefined): number | null {
  if (!raw) {
    return null
  }

  const pid = Number.parseInt(raw.trim().split(/\s+/)[0] ?? '', 10)

  return Number.isInteger(pid) && pid > 0 ? pid : null
}

/**
 * Operator-managed serve binds a real port. Desktop pool backends use
 * `--port 0` and are already stopped by the tracked-backend tree kill.
 */
export function isOperatorManagedServeCmdline(cmdline: string): boolean {
  if (!/hermes_cli\.main/i.test(cmdline) || !/(?:^|\s)serve(?:\s|$)/i.test(cmdline)) {
    return false
  }

  if (/(?:^|\s)--port(?:\s+|=)0(?:\s|$)/i.test(cmdline)) {
    return false
  }

  return /(?:^|\s)--port(?:\s+|=)\d+/i.test(cmdline) || /(?:^|\s)--host(?:\s+|=)\d+\.\d+\.\d+\.\d+/i.test(cmdline)
}

function isHermesOwnedUpdateCmdline(cmdline: string): boolean {
  return (
    /hindsight_api\.main/i.test(cmdline) ||
    /hermes_kernel_runner\.py/i.test(cmdline) ||
    /agent\.transports\.hermes_tools_mcp_server/i.test(cmdline) ||
    isOperatorManagedServeCmdline(cmdline)
  )
}

function exeIsInHermesInstall(exePath: string, venvScriptsDir: string, runtimePythonDir?: string): boolean {
  if (hasWindowsPathPrefix(exePath, venvScriptsDir)) {
    return true
  }

  return Boolean(runtimePythonDir && hasWindowsPathPrefix(exePath, runtimePythonDir))
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
 * Broader pre-update drain set: hindsight plus the always-on serve tree,
 * execute_code kernels, and venv-launched MCP bridges. Interpreters under
 * `.hermes-runtime\python\` are included because they (not the Scripts
 * trampoline) map `venv\Lib\site-packages\*.pyd`.
 */
export function isHermesOwnedUpdateHolder(
  exePath: string | null | undefined,
  cmdline: string | null | undefined,
  venvScriptsDir: string,
  runtimePythonDir?: string
): boolean {
  if (!exePath || !cmdline) {
    return false
  }

  return exeIsInHermesInstall(exePath, venvScriptsDir, runtimePythonDir) && isHermesOwnedUpdateCmdline(cmdline)
}
