'use strict'

/**
 * venv-blocker-scan.ts
 *
 * Thin helper that runs the Python venv-blocker scan as a subprocess and
 * returns a typed result for the Desktop update preflight.
 */

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

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
 * Run the venv-blocker scan subprocess.  Accepts optional overrides for
 * testing (dependency injection).
 */
export function scanVenvBlockers(
  updateRoot: string,
  execOverride?: typeof execFileSync,
  resolveOverride?: typeof resolveVenvPython,
): ScanOutcome {
  const execFn = execOverride || execFileSync
  const resolveFn = resolveOverride || resolveVenvPython
  const venvPython = resolveFn(updateRoot)

  if (!venvPython) {
    return { kind: 'probe-failure', error: 'venv python not found' }
  }

  let stdout: string

  try {
    const proc = execFn(
      venvPython,
      ['-m', SCAN_MODULE],
      {
        cwd: updateRoot,
        encoding: 'utf-8',
        stdio: ['ignore', 'pipe', 'pipe'],
        timeout: SCAN_TIMEOUT_MS,
        windowsHide: true,
      } as any,
    )

    stdout = (proc as unknown as string)
  } catch (err: any) {
    const diag = [`exit code ${err.status ?? -1}`]

    if (err.stderr) {diag.push(String(err.stderr).slice(0, 200))}

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
    '',
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
    'remote backend, stopping it will disconnect remote clients.',
  )
  lines.push(
    'Then retry the update.',
  )

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