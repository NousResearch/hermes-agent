/**
 * backend-claim.ts
 *
 * Process start-marker probes used to match backend ownership records against
 * live processes.
 */

import { execFile } from 'node:child_process'
import fs from 'node:fs'

import { electronProcessStartMarker } from './parent-process-identity'
import { isPidAlive } from './update-marker'
import { hiddenWindowsChildOptions } from './windows-child-options'

export function execText(command: string, args: string[], { timeout = 3000 } = {}): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const child = execFile(command, args, hiddenWindowsChildOptions({ encoding: 'utf8', timeout }), (error, stdout) => {
      if (error) {
        reject(error)
      } else if (timeout > 0 && child.killed) {
        // A SIGTERM handler can exit zero after execFile's timeout fired.
        reject(new Error(`${command} timed out after ${timeout}ms`))
      } else {
        resolve(String(stdout || '').trim())
      }
    })

    // These probes are noninteractive; do not leave readers waiting for input.
    child.stdin?.end()
  })
}

/**
 * Probe budget for the ORPHAN-REAP path (matchesParent / matchesIdentity /
 * stopOwnedBackend). The claim path keeps the full 30s headroom — a freshly
 * spawned backend's marker is load-bearing and a slow probe must not kill a
 * healthy child (#93608). Reap only needs to tell "same process" from "gone
 * or reused" for OLD records, and the ownership file can accumulate dozens of
 * them (one per profile per launch), so a 30s budget per record would let a
 * cold PowerShell 5.1 stall boot for minutes (#87169). 5s is plenty for a
 * warm probe; a timeout degrades to "unknown" and the record is preserved for
 * the next launch instead of blocking boot.
 */
export const REAP_PROBE_TIMEOUT_MS = 5_000

/**
 * Cross-platform process start marker: a value that changes when a PID is
 * reused, so `pid + marker` identifies one specific process incarnation.
 * Throws when the probe fails — callers decide what a failure means.
 */
export async function processStartMarker(pid: number, timeoutMs: number = 30_000): Promise<string> {
  // Cheap native dead-PID gate. Windows Get-Process / macOS `ps -p` exit 1
  // on a missing PID (not ESRCH), so the identity matchers used to keep the
  // orphan and re-probe it every launch (#92875). ESRCH is the code those
  // catch blocks already map to "gone". Alive or uninspectable (EPERM) PIDs
  // still fall through to the platform probe.
  if (!isPidAlive(pid)) {
    throw Object.assign(new Error(`PID ${pid} no longer exists`), { code: 'ESRCH' })
  }

  if (process.platform === 'linux') {
    const stat = await fs.promises.readFile(`/proc/${pid}/stat`, 'utf8')

    const fields = stat
      .slice(stat.lastIndexOf(')') + 1)
      .trim()
      .split(/\s+/)

    if (!/^\d+$/.test(fields[19] || '')) {
      throw new Error(`Invalid /proc start marker for PID ${pid}`)
    }

    return `linux:${fields[19]}`
  }

  if (process.platform === 'win32') {
    const electronMarker =
      pid === process.pid ? electronProcessStartMarker(pid, process.pid, process.getCreationTime?.()) : null

    if (electronMarker) {
      return electronMarker
    }

    const ticks = await execText(
      'powershell.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `$p = Get-Process -Id ${pid} -ErrorAction Stop; $p.StartTime.ToUniversalTime().Ticks`
      ],
      // PowerShell 5.1 cold starts routinely exceed the default 3s execText
      // budget (2.4-8s observed in #87169); give the marker probe headroom.
      // The claim path keeps this 30s budget; the orphan-reap path passes
      // REAP_PROBE_TIMEOUT_MS so a slow probe cannot stall boot.
      { timeout: timeoutMs }
    )

    if (!/^\d+$/.test(ticks)) {
      throw new Error(`Invalid Windows start marker for PID ${pid}`)
    }

    return `win:${ticks}`
  }

  const started = await execText('ps', ['-p', String(pid), '-o', 'lstart='])

  if (!started) {
    throw new Error(`Missing process start marker for PID ${pid}`)
  }

  return `ps:${started}`
}

const PID_ONLY_MARKER_PREFIX = 'pid-only:'

/**
 * Degraded identity marker written by the former local-spawn claim path when
 * the start-marker probe failed against a live child. Ownership records may
 * still carry it; identity matchers treat it as PID-liveness only.
 */
export function isPidOnlyStartMarker(startMarker: unknown): boolean {
  return typeof startMarker === 'string' && startMarker.startsWith(PID_ONLY_MARKER_PREFIX)
}
