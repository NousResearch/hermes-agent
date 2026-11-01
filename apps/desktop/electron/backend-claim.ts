/**
 * backend-claim.ts
 *
 * Process start-marker probes used to match backend ownership records against
 * live processes.
 */

import { execFile, spawn } from 'node:child_process'
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

const EXEC_TEXT_MAX_BUFFER = 1024 * 1024

/**
 * Run the SSH effective-config probe without allocating an input pipe. Some
 * Windows OpenSSH builds hang when Node starts `ssh -G` with every stream
 * piped, so this intentionally matches the stdin-ignored topology used by
 * the interactive SSH-config resolver below in main.ts.
 */
export function execTextWithIgnoredStdin(
  command: string,
  args: string[],
  { timeout = 3000, maxBuffer = EXEC_TEXT_MAX_BUFFER }: { timeout?: number; maxBuffer?: number } = {}
): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const child = spawn(command, args, hiddenWindowsChildOptions({ stdio: ['ignore', 'pipe', 'pipe'] }))
    let settled = false
    let stdout = ''
    let stderr = ''
    let stdoutBytes = 0
    let stderrBytes = 0
    let timer: ReturnType<typeof setTimeout> | undefined

    const settle = (callback: () => void) => {
      if (settled) {
        return
      }

      settled = true

      if (timer) {
        clearTimeout(timer)
      }

      callback()
    }

    const maxBufferError = () => {
      const error = Object.assign(new RangeError(`${command} output exceeded ${maxBuffer} bytes`), {
        code: 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER'
      })

      child.kill()
      settle(() => reject(error))
    }

    const collect = (stream: 'stdout' | 'stderr', chunk: unknown) => {
      if (settled) {
        return
      }

      const text = String(chunk)
      const bytes = Buffer.isBuffer(chunk) ? chunk.length : Buffer.byteLength(text)

      if (stream === 'stdout') {
        stdoutBytes += bytes
        stdout += text

        if (stdoutBytes > maxBuffer) {
          maxBufferError()
        }
      } else {
        stderrBytes += bytes
        stderr += text

        if (stderrBytes > maxBuffer) {
          maxBufferError()
        }
      }
    }

    child.stdout?.on('data', chunk => collect('stdout', chunk))
    child.stderr?.on('data', chunk => collect('stderr', chunk))
    child.once('error', error => settle(() => reject(error)))
    child.once('close', (code, signal) => {
      if (code === 0 && !child.killed) {
        settle(() => resolve(stdout.trim()))

        return
      }

      const error = Object.assign(new Error(stderr.trim() || `Command failed: ${command}`), { code, signal, stderr, stdout })
      settle(() => reject(error))
    })

    if (timeout > 0) {
      timer = setTimeout(() => {
        child.kill()
        settle(() => reject(new Error(`${command} timed out after ${timeout}ms`)))
      }, timeout)
    }
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
