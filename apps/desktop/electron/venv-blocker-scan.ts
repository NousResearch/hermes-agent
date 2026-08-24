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

export type VenvBlockerKind = 'local-preview' | 'other'

export interface VenvBlockerProcess {
  pid: number
  name: string
  cmdline: string
  kind: VenvBlockerKind
  safeToStop: boolean
  label?: string
  port?: number
  createTime?: number
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

// A Windows process table can take longer than 15 seconds to inspect on busy
// hosts. Keep a watchdog for genuinely wedged probes, but leave enough headroom
// for the scanner's conservative fallback checks.
const SCAN_TIMEOUT_MS = 60000
const SCAN_MODULE = 'hermes_cli._scan_venv_blockers'

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

function classifyVenvBlocker(
  process: Pick<VenvBlockerProcess, 'pid' | 'name' | 'cmdline'>,
  hints?: Record<string, unknown>
): VenvBlockerProcess {
  const moduleMatch = process.cmdline.match(/(?:^|\s)-m\s+http\.server(?:\s+(\d{1,5}))?(?:\s|$)/i)
  const isPython = /^python(?:w)?(?:\.exe)?$/i.test(process.name)
  const hintedCreateTime = typeof hints?.createTime === 'number' ? hints.createTime : undefined

  const trustedScannerIdentity =
    hints?.kind === 'local-preview' &&
    hints.safeToStop === true &&
    hintedCreateTime !== undefined &&
    Number.isFinite(hintedCreateTime) &&
    hintedCreateTime > 0

  if (!isPython || !moduleMatch || !trustedScannerIdentity) {
    return { ...process, kind: 'other', safeToStop: false }
  }

  const parsedPort = moduleMatch[1] ? Number(moduleMatch[1]) : 8000
  const hintedPort = trustedScannerIdentity && typeof hints?.port === 'number' ? hints.port : undefined
  const candidatePort = hintedPort ?? parsedPort

  const port =
    Number.isInteger(candidatePort) && candidatePort > 0 && candidatePort <= 65535 ? candidatePort : undefined

  const directoryMatch = process.cmdline.match(/(?:^|\s)--directory\s+(?:"([^"]+)"|'([^']+)'|(.+))$/i)
  const directory = (directoryMatch?.[1] || directoryMatch?.[2] || directoryMatch?.[3] || '').trim()
  const parsedLabel = directory ? path.win32.basename(directory.replace(/["']$/, '')) : undefined
  const hintedLabel = trustedScannerIdentity && typeof hints?.label === 'string' ? hints.label.trim() : ''
  const label = hintedLabel || parsedLabel

  return {
    ...process,
    kind: 'local-preview',
    safeToStop: true,
    ...(label ? { label } : {}),
    ...(port ? { port } : {}),
    createTime: hintedCreateTime
  }
}

/**
 * Stop only blockers that the fresh scanner identified as Python static-file
 * preview servers. Unknown Python/Hermes processes are deliberately ignored.
 */
export async function stopSafeVenvBlockers(
  updateRoot: string,
  result: VenvBlockerScanResult,
  execOverride?: typeof execFileAsync,
  resolvePython: typeof resolveVenvPython = resolveVenvPython
): Promise<{ stopped: number[]; failed: number[] }> {
  const execFn = execOverride || execFileAsync
  const stopped: number[] = []
  const failed: number[] = []
  const pythonPath = resolvePython(updateRoot)

  for (const process of result.processes) {
    if (
      !pythonPath ||
      !process.safeToStop ||
      process.kind !== 'local-preview' ||
      !process.createTime ||
      !Number.isFinite(process.createTime)
    ) {
      if (process.safeToStop && process.kind === 'local-preview') {
        failed.push(process.pid)
      }

      continue
    }

    try {
      await execFn(
        pythonPath,
        ['-m', 'hermes_cli._scan_venv_blockers', '--terminate-safe', String(process.pid), String(process.createTime)],
        { cwd: updateRoot, windowsHide: true, timeout: 10_000, maxBuffer: 256 * 1024 }
      )
      stopped.push(process.pid)
    } catch {
      failed.push(process.pid)
    }
  }

  return { stopped, failed }
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

    processes.push(classifyVenvBlocker({ pid, name, cmdline }, entry))
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
 * event loop is never blocked by the psutil process scan (tens of seconds on a
 * loaded Windows box).  Accepts optional overrides for testing (dependency
 * injection).
 */
export async function scanVenvBlockers(
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
    if (err?.killed === true) {
      return {
        kind: 'probe-failure',
        error: `timed out after ${SCAN_TIMEOUT_MS / 1000} seconds`
      }
    }

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
export function formatProbeFailedMessage(error?: string): string {
  const timeoutDetail = error?.startsWith('timed out after')
    ? `\n\nThe verification scan ${error}; no blocking process was confirmed.`
    : ''

  return (
    'Update aborted: Desktop could not verify the Hermes installation is free.' +
    timeoutDetail +
    '\n\n' +
    'Close other Hermes windows and terminals, then retry.  If the problem\n' +
    'persists, run `hermes update` in a terminal for detailed diagnostics.'
  )
}
