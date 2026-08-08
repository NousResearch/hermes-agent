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

/** Settling knobs for the blocked-verdict retry. Injectable for testing. */
export interface ScanSettleOptions {
  attempts?: number
  delayMs?: number
  sleep?: (ms: number) => Promise<void>
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SCAN_TIMEOUT_MS = 15000
const SCAN_MODULE = 'hermes_cli._scan_venv_blockers'

// The scan runs immediately after releaseBackendLock tree-kills the desktop's
// own backends. On Windows the OS does not retire those PIDs instantly —
// tree-killed grandchildren cascade-settle over several scheduler ticks, so
// psutil.process_iter() keeps reporting dying entries for a short window and
// the preflight aborts an update that was actually fine (#74805). Re-probe a
// couple of times before believing "blocked", mirroring the poll-until-clear
// pattern releaseBackendLock already uses for the venv shim.
const SCAN_BLOCKED_ATTEMPTS = 3
const SCAN_SETTLE_DELAY_MS = 500

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

function defaultSleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

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
 * Run the venv-blocker scan, re-probing a blocked verdict a few times so a
 * process table that has not finished retiring the backends we just killed
 * does not abort an otherwise-fine update (#74805).  Async so the Electron
 * main-process event loop is never blocked by the psutil process scan (up to
 * 15s per probe on a loaded Windows box).  Accepts optional overrides for
 * testing (dependency injection).
 */
export async function scanVenvBlockers(
  updateRoot: string,
  execOverride?: typeof execFileAsync,
  resolveOverride?: typeof resolveVenvPython,
  settleOverride?: ScanSettleOptions
): Promise<ScanOutcome> {
  const attempts = Math.max(1, settleOverride?.attempts ?? SCAN_BLOCKED_ATTEMPTS)
  const delayMs = Math.max(0, settleOverride?.delayMs ?? SCAN_SETTLE_DELAY_MS)
  const sleep = settleOverride?.sleep || defaultSleep

  let outcome = await runVenvBlockerProbe(updateRoot, execOverride, resolveOverride)

  // Only a 'blocked' verdict is retried. 'clear' is already the answer we
  // want, and 'probe-failure' means the venv state is unknown — re-running a
  // broken probe cannot turn that into knowledge, and the caller must abort
  // either way.
  for (let attempt = 1; attempt < attempts && outcome.kind === 'blocked'; attempt += 1) {
    await sleep(delayMs)
    outcome = await runVenvBlockerProbe(updateRoot, execOverride, resolveOverride)
  }

  return outcome
}

/**
 * One probe of the venv-blocker scan subprocess.  No retry, no settling delay
 * — callers wanting the race-tolerant behaviour should use scanVenvBlockers.
 */
export async function runVenvBlockerProbe(
  updateRoot: string,
  execOverride?: typeof execFileAsync,
  resolveOverride?: typeof resolveVenvPython
): Promise<ScanOutcome> {
  const execFn = execOverride || execFileAsync
  const resolveFn = resolveOverride || resolveVenvPython
  const venvPython = resolveFn(updateRoot)

  if (!venvPython) {
    return { kind: 'probe-failure', error: 'venv python not found' }
  }

  let stdout: string

  try {
    const proc = await execFn(venvPython, ['-m', SCAN_MODULE], {
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
