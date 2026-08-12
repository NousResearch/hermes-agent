'use strict'

/**
 * venv-blocker-scan.ts
 *
 * Thin helper that runs the Python venv-blocker scan as a subprocess and
 * returns a typed result for the Desktop update preflight.
 */

import { execFile } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface VenvBlockerProcess {
  pid: number
  name: string
  cmdline: string
}

export interface VenvBlockerScanResult {
  blocked: boolean
  processes: VenvBlockerProcess[]
}

export type ScanOutcome =
  | { kind: 'clear'; result: VenvBlockerScanResult }
  | { kind: 'blocked'; result: VenvBlockerScanResult }
  | { kind: 'probe-failure'; error: string }

/**
 * Options for the venv-blocker scan subprocess.
 */
export interface ScanOptions {
  /**
   * When set, all descendants of this PID are excluded from the scan.
   * Used by the Desktop app to skip its own respawned backends (#77277):
   * after `releaseBackendLockForUpdate` kills the backend, the Desktop
   * app respawns it within seconds — the scanner must not count those
   * as external blockers.
   */
  excludeChildrenOf?: number
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SCAN_TIMEOUT_MS = 15000
const SCAN_MODULE = 'hermes_cli._scan_venv_blockers'

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Strictly validate and parse the JSON output from the venv-blocker scan.
 * Pure function — no side effects.
 */
export function parseVenvBlockerScanOutput(raw: string): ScanOutcome {
  let parsed: any

  try {
    parsed = JSON.parse(raw)
  } catch {
    return { kind: 'probe-failure', error: 'malformed JSON' }
  }

  if (!parsed || typeof parsed !== 'object' || parsed.ok !== true) {
    return { kind: 'probe-failure', error: 'missing or invalid ok field' }
  }

  if (typeof parsed.blocked !== 'boolean') {
    return { kind: 'probe-failure', error: 'blocked must be a boolean' }
  }

  if (!Array.isArray(parsed.processes)) {
    return { kind: 'probe-failure', error: 'processes must be an array' }
  }

  const processes: VenvBlockerProcess[] = []

  for (const entry of parsed.processes) {
    if (!entry || typeof entry !== 'object') {
      return { kind: 'probe-failure', error: 'process entry must be an object' }
    }

    const { pid, name, cmdline } = entry

    if (!Number.isInteger(pid) || pid <= 0) {
      return { kind: 'probe-failure', error: 'process pid must be a positive integer' }
    }

    if (typeof name !== 'string' || name.length === 0) {
      return { kind: 'probe-failure', error: 'process name must be a non-empty string' }
    }

    if (typeof cmdline !== 'string') {
      return { kind: 'probe-failure', error: 'process cmdline must be a string' }
    }

    processes.push({ pid, name, cmdline })
  }

  // Reject inconsistent combinations
  if (parsed.blocked && processes.length === 0) {
    return { kind: 'probe-failure', error: 'blocked is true but process list is empty' }
  }

  if (!parsed.blocked && processes.length > 0) {
    return { kind: 'probe-failure', error: 'blocked is false but process list is non-empty' }
  }

  return parsed.blocked
    ? { kind: 'blocked', result: { blocked: true, processes } }
    : { kind: 'clear', result: { blocked: false, processes } }
}

/**
 * Run the venv-blocker scan subprocess.  Async so the Electron main-process
 * event loop is never blocked by the psutil process scan (up to 15s on a
 * loaded Windows box.  Accepts optional overrides for testing (dependency
 * injection).
 *
 * @param updateRoot - The hermes-agent install root.
 * @param execOverride - Optional exec override for testing.
 * @param resolveOverride - Optional resolve override for testing.
 * @param options - Scan options (e.g. excludeChildrenOf for Desktop self-skip).
 */
export async function scanVenvBlockers(
  updateRoot: string,
  execOverride?: typeof execFileAsync,
  resolveOverride?: typeof resolveVenvPython,
  options?: ScanOptions
): Promise<ScanOutcome> {
  const execFn = execOverride || execFileAsync
  const resolveFn = resolveOverride || resolveVenvPython
  const venvPython = resolveFn(updateRoot)

  if (!venvPython) {
    return { kind: 'probe-failure', error: 'venv python not found' }
  }

  // Build the argument list.  When the Desktop app passes its own PID,
  // the scanner excludes all of that PID's descendants — preventing
  // respawned backends from being reported as blockers (#77277).
  const scanArgs = ['-m', SCAN_MODULE]

  if (options?.excludeChildrenOf != null && Number.isInteger(options.excludeChildrenOf)) {
    scanArgs.push('--exclude-children-of', String(options.excludeChildrenOf))
  }

  let stdout: string

  try {
    const proc = await execFn(venvPython, scanArgs, {
      cwd: updateRoot,
      encoding: 'utf-8',
      timeout: SCAN_TIMEOUT_MS,
      windowsHide: true
    } as any)

    stdout = String((proc as any).stdout ?? '')
  } catch (err: any) {
    const diag = [`exit code ${err.status ?? err.code ?? -1}`]

    if (err.stderr) {
      diag.push(String(err.stderr).slice(0, 200))
    }

    return { kind: 'probe-failure', error: diag.join('; ') }
  }

  return parseVenvBlockerScanOutput(stdout)
}

// ---------------------------------------------------------------------------
// Internal helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Resolve the venv python path.  Returns null if the file does not exist. */
export function resolveVenvPython(updateRoot: string): string | null {
  const isWindows = process.platform === 'win32'
  const pythonName = isWindows ? 'python.exe' : 'python3'
  const scriptsDir = isWindows ? 'Scripts' : 'bin'
  const candidate = path.join(updateRoot, 'venv', scriptsDir, pythonName)

  try {
    fs.accessSync(candidate)

    return candidate
  } catch {
    return null
  }
}

/**
 * Build a human-readable error message from blocker scan results.
 * Does NOT recommend --force-venv.
 */
export function formatBlockerMessage(result: VenvBlockerScanResult): string {
  const lines = [
    'Update aborted: another Hermes process is using this installation.',
    '',
    'These processes must be stopped before updating:',
    ''
  ]

  for (const proc of result.processes.slice(0, 10)) {
    lines.push(`  PID ${proc.pid}  ${proc.name}  ${proc.cmdline}`)
  }

  if (result.processes.length > 10) {
    lines.push(`  ... and ${result.processes.length - 10} more`)
  }

  lines.push('')
  lines.push(
    'Close the terminal, app, or service owning that process.  If it is a ' +
      'remote backend, stopping it will disconnect remote clients.'
  )
  lines.push('Then retry the update.')

  return lines.join('\n')
}

/**
 * Build a probe-failure error message.
 */
export function formatProbeFailedMessage(): string {
  return (
    'Update aborted: Desktop could not verify the Hermes installation is free.\n' +
    '\n' +
    'Close other Hermes windows and terminals, then retry.  If the problem\n' +
    'persists, run `hermes update` in a terminal for detailed diagnostics.'
  )
}
