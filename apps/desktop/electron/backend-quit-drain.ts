/**
 * Backend quit-drain decision logic (pure — no Electron side effects here).
 *
 * The desktop's `before-quit` handler must wait for the hermes backend
 * processes to actually exit before Electron tears down, or the SIGTERM races
 * ahead and leaves stray `hermes serve` orphans. The one trap: the child
 * handles must be captured BEFORE `stopAllPoolBackends()` runs, because it
 * deletes every backendPool entry (and thus the reference to each pool child)
 * before sending SIGTERM.
 *
 * This module exposes the pure computation so the drain behavior is unit-testable
 * without booting Electron.
 */

/**
 * A handle to a backend child process. Only `exitCode`/`signalCode`/`killed`
 * are read here, so a real ChildProcess or a minimal test stub both work.
 */
export interface BackendChildLike {
  exitCode: number | null
  signalCode: NodeJS.Signals | null
  killed?: boolean
}

/**
 * Compute the set of backend children that must be waited on during quit.
 *
 * @param primaryProcess The primary (window) backend process handle, if any.
 * @param poolEntries    A snapshot of the current backendPool entries
 *                       (profile -> { process, ... }). Callers MUST pass this
 *                       before calling stopAllPoolBackends(), since that
 *                       clears the pool.
 * @returns A deduplicated list of still-alive children to drain.
 */
export function collectBackendDrainTargets(
  primaryProcess: BackendChildLike | null | undefined,
  poolEntries: ReadonlyArray<{ process: BackendChildLike | null | undefined }>,
): BackendChildLike[] {
  const seen = new Set<BackendChildLike>()
  const out: BackendChildLike[] = []

  for (const child of [primaryProcess, ...poolEntries.map(e => e.process)]) {
    if (!child || seen.has(child)) {
      continue
    }

    seen.add(child)

    // Already-exited handles need no waiting; only live ones are targets.
    if (child.exitCode === null && child.signalCode === null) {
      out.push(child)
    }
  }

  return out
}
