'use strict'

/**
 * venv-orphan-reap.ts
 *
 * Find orphaned Hermes processes that hold THIS install's venv shim open, so
 * the update hand-off can reap them instead of aborting forever.
 *
 * Why this exists: `releaseBackendLock` in main.ts tree-kills only the PIDs the
 * CURRENT desktop instance owns (its primary backend plus the backend pool),
 * then polls `venv/Scripts/hermes.exe` for up to 15s. A `hermes gateway run` or
 * `hermes dashboard` left behind by a PREVIOUS desktop session is not in either
 * registry, so nothing ever kills it. It keeps the shim mandatory-locked, the
 * poll times out, and every future update aborts with "another process is
 * holding the Hermes install open ... Close it and retry" — pointing at a
 * window that no longer exists. The install is wedged with no route forward
 * short of Task Manager. Observed on 2026-08-20 and 2026-08-21 with two such
 * orphans (`hermes dashboard --port 9119` and `hermes gateway run`), both with
 * dead parents.
 *
 * This does NOT overlap `backendOwnership.reapOrphans()`, which reaps entries
 * recorded in the desktop's own ownership file and therefore only ever sees
 * backends the desktop itself spawned. Neither observed holder qualified: the
 * gateway was started from a user terminal that has since exited, and
 * `backendCommandMatches` only recognises `serve` and `dashboard`, so
 * `hermes gateway run` is not a "backend" by that definition at all. Scanning
 * for the shim path catches a holder whatever spawned it.
 *
 * Scope is deliberately narrow on two axes:
 *
 *  - EXACT shim path, so a second Hermes install on the same machine is never
 *    touched.
 *  - ORPHANS ONLY (parent PID no longer present). A `hermes` a user is running
 *    in their own terminal has that terminal as a live parent; they can see it
 *    and close it, which is exactly what the existing error message asks for.
 *    Killing it out from under them would destroy interactive work nobody
 *    asked us to end. An orphan has no such owner — reaping it is the only way
 *    the update can proceed.
 *
 * PID reuse can make a dead parent look alive, which suppresses one reap and
 * costs the user a retry. That is the safe direction to fail.
 */

export interface VenvProcessSnapshot {
  pid: number
  parentPid: number
  executablePath?: string | null
}

export interface FindOrphanedVenvHoldersDeps {
  isWindows?: boolean
  selfPid?: number
  execText?: (command: string, args: string[]) => Promise<string>
  /** Records why the probe produced nothing, so a silent miss is diagnosable. */
  onProbeFailure?: (message: string) => void
}

/**
 * Pure selection: which snapshot rows are orphaned holders of `shimPath`.
 *
 * `snapshot` must be the FULL process list — parent liveness is decided by
 * whether the parent PID appears in it, so a filtered list would report every
 * match as an orphan.
 */
export function selectOrphanedVenvHolders(
  snapshot: readonly VenvProcessSnapshot[],
  shimPath: string,
  selfPid: number
): number[] {
  const target = shimPath.toLowerCase()
  const livePids = new Set(snapshot.map(entry => entry.pid))

  return snapshot
    .filter(entry => {
      if (entry.pid === selfPid) {
        return false
      }

      if ((entry.executablePath || '').toLowerCase() !== target) {
        return false
      }

      // parentPid 0 is "no parent" on Windows (the row Task Manager shows as
      // System Idle), so it can never be a live owner.
      return entry.parentPid === 0 || !livePids.has(entry.parentPid)
    })
    .map(entry => entry.pid)
}

/**
 * Parse the JSON snapshot emitted by the PowerShell probe.
 *
 * Windows PowerShell 5.1's ConvertTo-Json collapses a one-element collection
 * to a bare object, so both shapes must be accepted — assuming an array turns
 * "exactly one orphan" (the common case) into "no orphans found".
 *
 * Never throws: the caller is already inside a failing update, and a parse
 * error must degrade to the pre-existing behavior, not add a new abort.
 */
export function parseProcessSnapshot(raw: string): VenvProcessSnapshot[] {
  const text = String(raw || '').trim()

  if (!text) {
    return []
  }

  let parsed: unknown

  try {
    parsed = JSON.parse(text)
  } catch {
    return []
  }

  const rows = Array.isArray(parsed) ? parsed : [parsed]
  const snapshot: VenvProcessSnapshot[] = []

  for (const row of rows) {
    if (!row || typeof row !== 'object') {
      continue
    }

    const { ProcessId, ParentProcessId, ExecutablePath } = row as Record<string, unknown>

    if (!Number.isInteger(ProcessId) || (ProcessId as number) <= 0) {
      continue
    }

    snapshot.push({
      pid: ProcessId as number,
      parentPid: Number.isInteger(ParentProcessId) ? (ParentProcessId as number) : 0,
      executablePath: typeof ExecutablePath === 'string' ? ExecutablePath : null
    })
  }

  return snapshot
}

// Get-CimInstance Win32_Process measured 2.5s on a 383-process box, and
// main.ts already documents 2.4-8s for the same call under load (#87169) with
// a 30s budget. A tight budget here would turn a busy machine into a silent
// no-reap and put the update straight back into the wedge this file exists to
// clear, so match that headroom — the caller's own 15s poll is what bounds the
// wait in practice.
const PROBE_TIMEOUT_MS = 30_000

/**
 * Probe the live process table and return orphaned holders of `shimPath`.
 *
 * Best-effort by contract: any probe failure returns an empty list, leaving
 * the caller's existing owned-PID sweep exactly as it was.
 */
export async function findOrphanedVenvHolders(
  shimPath: string,
  deps: FindOrphanedVenvHoldersDeps = {}
): Promise<number[]> {
  const isWindows = deps.isWindows ?? process.platform === 'win32'

  // POSIX has no mandatory file locks, so a running shim never blocks the
  // update and there is nothing to reap.
  if (!isWindows) {
    return []
  }

  const selfPid = deps.selfPid ?? process.pid
  const execText = deps.execText
  const onProbeFailure = deps.onProbeFailure ?? (() => {})

  if (!execText) {
    onProbeFailure('no execText provided; skipping orphan scan')

    return []
  }

  let raw: string

  try {
    raw = await execText('powershell.exe', [
      '-NoProfile',
      '-NonInteractive',
      '-Command',
      'Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, ExecutablePath | ConvertTo-Json -Compress'
    ])
  } catch (error) {
    onProbeFailure(`process probe failed: ${(error as Error)?.message || 'unknown error'}`)

    return []
  }

  const snapshot = parseProcessSnapshot(raw)

  // An empty snapshot from a non-empty probe means the output shape changed,
  // not that the machine has no processes. Say so rather than reporting a
  // clean "no orphans" that sends the caller back into a 15s timeout.
  if (snapshot.length === 0) {
    onProbeFailure(`process probe returned no usable rows (${String(raw || '').length} bytes)`)

    return []
  }

  return selectOrphanedVenvHolders(snapshot, shimPath, selfPid)
}

export { PROBE_TIMEOUT_MS }
