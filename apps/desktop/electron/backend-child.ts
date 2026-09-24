/**
 * backend-child.ts
 *
 * Windows-aware teardown for the desktop's managed backend child process.
 *
 * Node's `child.kill()` only signals the direct child. On Windows a backend
 * that spawned its own grandchildren (a `hermes` REPL, a pty terminal
 * session, the gateway) survives a plain SIGTERM and keeps files (e.g. the
 * venv shim) locked. So on Windows we tree-kill via `forceKillProcessTree`.
 *
 * On POSIX the backend IS spawned into its own session/process-group
 * (start_new_session=True), so `child.kill('SIGTERM')` would only reach the
 * backend and orphan its MCP grandchildren (the leak in #serve-orphans). We
 * signal the whole group via `process.kill(-pid, ...)` instead, falling back
 * to the direct child if the group send fails.
 *
 * Extracted into its own dependency-free module (no electron import) so the
 * tree-kill / group-kill branching can be asserted directly with a fake child
 * object and spy kill functions, instead of grepping main.ts source text for
 * the function body.
 */

export interface StopBackendChildDeps {
  /** Defaults to the real platform check; injectable for tests. */
  isWindows?: boolean
  /**
   * Windows tree-kill implementation (real: taskkill /T /F via execFileSync).
   * Receives the threaded kill reason so the supervisor can log pid+reason
   * before the kill runs (issue #119440 attribution).
   */
  forceKillProcessTree: (pid: number, reason?: BackendTreeKillReason) => void
  /**
   * POSIX group-signal implementation. Real: process.kill(-pgid, signal).
   * Injectable so the negative-pid group send is asserted in tests without a
   * live process group. Defaults to process.kill.
   */
  killGroup?: (pgid: number, signal: string) => void
}

export interface StopBackendTreesForUpdateDeps {
  /** Synchronous Windows taskkill /T /F implementation. */
  forceKillProcessTree: (pid: number, reason?: BackendTreeKillReason) => void
  /** Clears and stops the desktop's pooled backends. */
  stopAllPoolBackends: () => void
}

/**
 * Why a backend process tree is being killed (#119440). Threaded from every
 * stop path to the tree-kill so a later abrupt `-1` / `4294967295` exit with
 * empty stderr names its killer instead of respawning blind.
 */
export type BackendTreeKillReason =
  | 'quit'
  | 'update-handoff'
  | 'superseded-start'
  | 'pool-stop'
  | 'orphan-reap'
  | 'stop-escalation'
  | 'unknown'

export interface StopBackendChildOptions {
  /** Why this child is being stopped; recorded with the tree-kill. */
  reason?: BackendTreeKillReason
}

/** Single log line the supervisor emits before taskkill /T /F runs. */
export function formatTreeKillLine(pid: number, reason: BackendTreeKillReason): string {
  return `[backend-child] tree-kill pid=${pid} reason=${reason}`
}

export interface BackendProcessRoot {
  pid?: number | null
}

export interface KillableChild extends BackendProcessRoot {
  killed?: boolean
  kill: (signal: NodeJS.Signals) => void
}

export interface WaitableChild extends KillableChild {
  exitCode: number | null
  signalCode: string | null
  once: (event: 'exit' | 'error', listener: () => void) => unknown
  removeListener: (event: 'exit' | 'error', listener: () => void) => unknown
}

/** Graceful exit, SIGKILL escalation, then a bounded wait for the escalation. */
export async function waitForBackendExit(
  child: WaitableChild | null | undefined,
  deps: StopBackendChildDeps,
  timeoutMs = 5000
): Promise<void> {
  if (!child || child.exitCode !== null || child.signalCode !== null) {
    return
  }

  const exited = () => child.exitCode !== null || child.signalCode !== null

  const wait = (delay: number) =>
    new Promise<void>(resolve => {
      if (exited()) {
        resolve()

        return
      }

      const finish = () => {
        clearTimeout(timer)
        child.removeListener('exit', finish)
        resolve()
      }

      const timer = setTimeout(finish, delay)
      child.once('exit', finish)
    })

  await wait(timeoutMs)

  if (exited()) {
    return
  }

  try {
    if ((deps.isWindows ?? process.platform === 'win32') && Number.isInteger(child.pid)) {
      deps.forceKillProcessTree(child.pid as number, 'stop-escalation')
    } else if (Number.isInteger(child.pid)) {
      try {
        const killGroup = deps.killGroup ?? ((pid, signal) => process.kill(pid, signal))
        killGroup(-(child.pid as number), 'SIGKILL')
      } catch {
        child.kill('SIGKILL')
      }
    } else {
      child.kill('SIGKILL')
    }
  } catch {
    // A failed signal may mean the child is gone, but only exit proves it.
  }

  await wait(1000)

  if (!exited()) {
    throw new Error(
      `Backend child${child.pid ? ` (PID ${child.pid})` : ''} did not exit after SIGKILL; retaining ownership.`
    )
  }
}

/**
 * Stop a managed child process, choosing the right strategy for the platform.
 * No-ops silently if `child` is falsy, already killed, or the kill attempt
 * throws (the process may already be gone) -- mirrors the original inline
 * best-effort semantics in main.ts.
 */
export function stopBackendChild(
  child: KillableChild | null | undefined,
  deps: StopBackendChildDeps,
  options: StopBackendChildOptions = {}
) {
  if (!child || child.killed) {
    return
  }

  const reason = options.reason ?? 'unknown'
  const isWindows = deps.isWindows ?? process.platform === 'win32'
  const killGroup = deps.killGroup ?? ((pgid: number, signal: string) => process.kill(pgid, signal))

  try {
    if (isWindows && Number.isInteger(child.pid)) {
      deps.forceKillProcessTree(child.pid as number, reason)
    } else if (Number.isInteger(child.pid)) {
      // POSIX: pgid == pid (start_new_session). Signal the whole group so MCP
      // grandchildren die too; fall back to the direct child on failure.
      try {
        killGroup(-(child.pid as number), 'SIGTERM')
      } catch {
        child.kill('SIGTERM')
      }
    } else {
      child.kill('SIGTERM')
    }
  } catch {
    // Already gone.
  }
}

/**
 * Stop every backend tree owned by a Windows Desktop update hand-off.
 *
 * Tree-kill the primary root while its PID is still live, then delegate pool
 * teardown to the existing routine that tree-kills each pooled root exactly
 * once before mutating its registry. In particular, do not signal the primary
 * first: if that root exits before taskkill /T runs, Windows can no longer
 * enumerate its MCP grandchildren and they survive with the venv locked.
 */
export function stopBackendTreesForUpdate(
  primary: BackendProcessRoot | null | undefined,
  deps: StopBackendTreesForUpdateDeps
): void {
  if (primary && Number.isInteger(primary.pid)) {
    deps.forceKillProcessTree(primary.pid as number, 'update-handoff')
  }

  deps.stopAllPoolBackends()
}

/**
 * Ledger of this process's own tree-kills (#119440). When a backend child
 * later exits `-1` / `4294967295` with empty stderr, the exit handler
 * consults this ledger to say whether THIS process killed it (and why) or
 * whether an external killer / sibling instance is implicated.
 */
export interface TreeKillRecord {
  pid: number
  reason: BackendTreeKillReason
  at: number
}

/** How far back an exit attributes to our own tree-kill. */
export const TREE_KILL_ATTRIBUTION_WINDOW_MS = 15_000

const MAX_TREE_KILL_RECORDS = 20

const treeKillRecords: TreeKillRecord[] = []

export function noteTreeKill(pid: number, reason: BackendTreeKillReason, now: number = Date.now()): void {
  treeKillRecords.push({ at: now, pid, reason })

  while (treeKillRecords.length > MAX_TREE_KILL_RECORDS) {
    treeKillRecords.shift()
  }
}

export function describeRecentTreeKills(
  now: number = Date.now(),
  windowMs: number = TREE_KILL_ATTRIBUTION_WINDOW_MS
): string {
  const recent = treeKillRecords.filter(record => now - record.at >= 0 && now - record.at <= windowMs)

  if (recent.length === 0) {
    return `no tree-kill by this process in the last ${Math.round(windowMs / 1000)}s`
  }

  return `recent tree-kill by this process: ${recent.map(record => `pid=${record.pid} reason=${record.reason}`).join(', ')}`
}

/**
 * True for the abrupt TerminateProcess-class death in #119440: Node reports
 * the Win32 DWORD `0xFFFFFFFF` (4294967295) or `-1`, with no signal and
 * (typically) empty stderr — the signature of taskkill /T /F, whether ours
 * or an external killer's.
 */
export function isAbruptWindowsKillExit(code: number | null, signal: string | null): boolean {
  return signal == null && (code === -1 || code === 4294967295)
}

/**
 * Log suffix for a backend exit line. Silent for ordinary exits; for an
 * abrupt `-1` / `4294967295` exit it carries the ownership snippet and this
 * process's recent tree-kill ledger so the death is attributable.
 */
export function describeAbruptBackendExit(args: {
  code: number | null
  signal?: string | null
  ownerText: string
  recentText: string
}): string {
  if (!isAbruptWindowsKillExit(args.code, args.signal ?? null)) {
    return ''
  }

  return ` [abrupt TerminateProcess-class death (empty stderr is typical of taskkill /T /F); ${args.ownerText}; ${args.recentText}]`
}
