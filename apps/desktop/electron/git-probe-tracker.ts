/**
 * Tracks outstanding git child processes spawned for passive probes (the
 * bundle-skew version check) so they can be killed at app quit.
 *
 * A treeless (`tree:0`) checkout can make a path-limited `rev-list`
 * lazy-fetch missing trees from origin, which makes git itself spawn
 * `fetch -> index-pack -> pack-objects` as ITS OWN children. Killing only the
 * top-level PID does not reap that chain: it survives an app restart,
 * reparented to PID 1 (#125243). Killing goes through the same
 * platform-aware strategy as the managed backend child (`stopBackendChild` in
 * backend-child.ts): Windows tree-kill by PID ancestry, POSIX group-signal by
 * negative pgid (the probe must be spawned `detached: true` so it leads its
 * own process group and that signal reaches its git-spawned descendants too).
 *
 * Dependency-injected and Electron-free so it is unit-testable directly.
 */

import { type KillableChild, stopBackendChild, type StopBackendChildDeps } from './backend-child'

/** A ChildProcess narrowed to what tracking + stopBackendChild() need. */
export interface TrackedGitChild extends KillableChild {
  once: (event: 'close' | 'error', listener: (arg?: unknown) => void) => unknown
}

export interface GitProbeTracker {
  /** Register a freshly spawned probe child; stops tracking it once it exits. */
  track: (child: TrackedGitChild) => void
  /** Kill every still-running tracked child (and its descendant tree). */
  killAll: () => void
  size: () => number
}

export function createGitProbeTracker(deps: StopBackendChildDeps): GitProbeTracker {
  const children = new Set<TrackedGitChild>()

  function track(child: TrackedGitChild): void {
    children.add(child)
    const stopTracking = () => children.delete(child)

    child.once('error', stopTracking)
    child.once('close', stopTracking)
  }

  function killAll(): void {
    for (const child of children) {
      stopBackendChild(child, deps)
    }
  }

  return { track, killAll, size: () => children.size }
}
