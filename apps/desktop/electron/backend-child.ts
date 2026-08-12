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
  /** Windows tree-kill implementation (real: taskkill /T /F via execFileSync). */
  forceKillProcessTree: (pid: number) => void
  /**
   * POSIX group-signal implementation. Real: process.kill(-pgid, signal).
   * Injectable so the negative-pid group send is asserted in tests without a
   * live process group. Defaults to process.kill.
   */
  killGroup?: (pgid: number, signal: string) => void
}

export interface StopBackendTreesForUpdateDeps {
  /** Synchronous Windows taskkill /T /F implementation. */
  forceKillProcessTree: (pid: number) => void
  /** Clears and stops the desktop's pooled backends. */
  stopAllPoolBackends: () => void
}

export interface BackendProcessRoot {
  pid?: number | null
}

export interface KillableChild extends BackendProcessRoot {
  killed?: boolean
  kill: (signal: string) => void
}

/**
 * Stop a managed child process, choosing the right strategy for the platform.
 * No-ops silently if `child` is falsy, already killed, or the kill attempt
 * throws (the process may already be gone) -- mirrors the original inline
 * best-effort semantics in main.ts.
 */
export function stopBackendChild(child: KillableChild | null | undefined, deps: StopBackendChildDeps) {
  if (!child || child.killed) {
    return
  }

  const isWindows = deps.isWindows ?? process.platform === 'win32'
  const killGroup = deps.killGroup ?? ((pgid: number, signal: string) => process.kill(pgid, signal))

  try {
    if (isWindows && Number.isInteger(child.pid)) {
      deps.forceKillProcessTree(child.pid as number)
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
    deps.forceKillProcessTree(primary.pid as number)
  }

  deps.stopAllPoolBackends()
}
