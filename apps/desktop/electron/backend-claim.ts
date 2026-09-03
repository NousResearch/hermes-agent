/**
 * backend-claim.ts
 *
 * The start-marker probe and the claim decision for a freshly spawned local
 * backend child, extracted from main.ts so the policy is testable without
 * booting Electron — including on a Windows CI lane that drives the probe
 * with REAL PowerShell (`processStartMarker` shells out to powershell.exe).
 *
 * Why this exists (#93608): `claimBackendChild` used to hard-fail on ANY
 * probe error — `Get-Process` timing out on a PowerShell 5.1 cold start
 * (see #87169) killed a perfectly healthy backend, the renderer "repaired"
 * by respawning, and the next probe timeout killed that one too. The rule is
 * now the same one `createParentStartMarkerResolver` already applies to the
 * parent marker: a failed probe against a LIVE child degrades to PID-only
 * identity instead of killing the child; only a child that actually DIED
 * keeps the fail-closed throw (now carrying its stderr tail, so the real
 * exit reason reaches desktop.log and the boot UI).
 */

import { execFile } from 'node:child_process'
import fs from 'node:fs'

import { electronProcessStartMarker } from './parent-process-identity'
import { isPidAlive } from './update-marker'
import { hiddenWindowsChildOptions } from './windows-child-options'

export function execText(command: string, args: string[], { timeout = 3000 } = {}): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    execFile(command, args, hiddenWindowsChildOptions({ encoding: 'utf8', timeout }), (error, stdout) => {
      if (error) {
        reject(error)
      } else {
        resolve(String(stdout || '').trim())
      }
    })
  })
}

/** Cheap cross-platform existence probe used before an expensive start-marker
 * lookup. On Windows a cold PowerShell process can take seconds, so running one
 * for every long-dead ownership record made Desktop startup scale with the
 * history of prior backends. `undefined` preserves fail-closed behavior for
 * unexpected permission/platform errors. */
export function processLiveness(
  pid: number,
  signal: (pid: number, signal: 0) => void = (target, value) => process.kill(target, value)
): boolean | undefined {
  try {
    signal(pid, 0)

    return true
  } catch (error) {
    const code = (error as NodeJS.ErrnoException | null)?.code

    if (code === 'ESRCH' || code === 'ENOENT') {
      return false
    }

    if (code === 'EPERM') {
      return true
    }

    return undefined
  }
}

/**
 * Cross-platform process start marker: a value that changes when a PID is
 * reused, so `pid + marker` identifies one specific process incarnation.
 * Throws when the probe fails — callers decide what a failure means (see
 * `claimDecision` / `probeStartMarker`).
 */
export async function processStartMarker(pid: number): Promise<string> {
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
      { timeout: 30_000 }
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

export type StartMarkerProbe = { ok: true; startMarker: string } | { ok: false; reason: string }

/** Run the marker probe, converting a throw into a value the pure decision can consume. */
export async function probeStartMarker(
  pid: number,
  probe: (pid: number) => Promise<string> = processStartMarker
): Promise<StartMarkerProbe> {
  try {
    return { ok: true, startMarker: await probe(pid) }
  } catch (error) {
    return { ok: false, reason: error instanceof Error ? error.message : String(error) }
  }
}

const PID_ONLY_MARKER_PREFIX = 'pid-only:'

/**
 * Degraded identity marker recorded when the start-marker probe failed but
 * the child was verifiably alive. It satisfies the ownership schema (a
 * non-empty startMarker) while telling identity matchers that only PID
 * liveness — plus the command-line check layered on top — can be verified.
 */
export function pidOnlyStartMarker(pid: number): string {
  return `${PID_ONLY_MARKER_PREFIX}${pid}`
}

export function isPidOnlyStartMarker(startMarker: unknown): boolean {
  return typeof startMarker === 'string' && startMarker.startsWith(PID_ONLY_MARKER_PREFIX)
}

export type ClaimDecision =
  { action: 'claim'; startMarker: string } | { action: 'degrade'; reason: string } | { action: 'fail'; reason: string }

/**
 * Pure claim policy for a freshly spawned backend child:
 *
 * - probe succeeded            → claim with the full start marker (unchanged).
 * - probe failed, child ALIVE  → degrade to PID-only identity; NEVER kill a
 *                                healthy backend over a flaky identity probe.
 * - probe failed, child DEAD   → fail closed; the child's death is the real
 *                                story and the caller attaches its stderr tail.
 */
export function claimDecision(childAlive: boolean, probe: StartMarkerProbe): ClaimDecision {
  if (probe.ok === true) {
    return { action: 'claim', startMarker: probe.startMarker }
  }

  const { reason } = probe

  return childAlive ? { action: 'degrade', reason } : { action: 'fail', reason }
}

export interface BackendOutputTail {
  /** Attach stdout/stderr data listeners to a just-spawned child. */
  attach(child: {
    stdout?: { on: (event: 'data', listener: (chunk: unknown) => void) => unknown } | null
    stderr?: { on: (event: 'data', listener: (chunk: unknown) => void) => unknown } | null
  }): void
  append(chunk: unknown): void
  /** The buffered tail (most recent `limit` characters), or ''. */
  text(): string
  /** Human-readable suffix for error messages, or '' when nothing buffered. */
  describe(): string
}

export const DEFAULT_OUTPUT_TAIL_LIMIT = 8192

/**
 * Ring-buffered tail of a child's combined stdout+stderr, attached at SPAWN
 * time — before the claim, before the READY wait — so an early crash's real
 * stderr (traceback, missing module, bad config) survives into the ownership
 * error and the before-ready exit messages instead of a bare exit code.
 */
export function createBackendOutputTail(limit: number = DEFAULT_OUTPUT_TAIL_LIMIT): BackendOutputTail {
  let buffer = ''

  const append = (chunk: unknown) => {
    buffer += String(chunk)

    if (buffer.length > limit) {
      buffer = buffer.slice(buffer.length - limit)
    }
  }

  return {
    append,
    attach(child) {
      child.stdout?.on('data', append)
      child.stderr?.on('data', append)
    },
    text() {
      return buffer
    },
    describe() {
      const text = buffer.trim()

      return text ? `\nRecent backend output:\n${text}` : ''
    }
  }
}
