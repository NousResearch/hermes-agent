/**
 * Windows Desktop close/stop after the existing tree-kill.
 *
 * taskkill /T /F is not widened: the caller passes the owned PIDs and the
 * same tree-kill. This module does not discover extra processes to signal.
 * A taskkill error is recorded, never discarded. After the kill, owned PIDs
 * are inventoried, and only locks with no live holder are cleared.
 */

export interface RuntimeLock {
  path: string
  /** PIDs named by the lock file, if any. */
  holderPids: number[]
  /**
   * Explicit hold (Restart Manager / open handle). True means a live process
   * holds the file even if we do not have its PID. Omit when unknown.
   */
  held?: boolean
}

export interface TaskkillFailure {
  pid: number
  error: string
}

export interface CloseStopKillDeps {
  /** Existing tree-kill (taskkill /PID n /T /F). Must not be widened by this module. */
  killTree: (pid: number) => void
  /** True while `pid` is still in the process table. */
  isPidAlive: (pid: number) => boolean
  clearLock: (path: string) => void
}

export interface CloseStopKillResult {
  taskkillFailures: TaskkillFailure[]
  /** Owned PIDs still enumerable after the tree kill. */
  remainingPids: number[]
  clearedLocks: string[]
  retainedLocks: string[]
  /**
   * True when a taskkill failure left an owned PID alive, or any owned PID
   * is still enumerable. An already-gone taskkill error is recorded but is
   * not itself a live failure.
   */
  liveFailure: boolean
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function lockIsHeld(lock: RuntimeLock, alive: (pid: number) => boolean): boolean {
  if (lock.held === true) {
    return true
  }

  return lock.holderPids.some(pid => Number.isInteger(pid) && pid > 0 && alive(pid))
}

/**
 * Tree-kill the owned PIDs, inventory what is still alive, and clear locks
 * that no live holder owns. Does not signal any PID that was not passed in.
 */
export function finishWindowsCloseStop(
  ownedPids: number[],
  locks: RuntimeLock[],
  deps: CloseStopKillDeps
): CloseStopKillResult {
  const pids = ownedPids.filter(pid => Number.isInteger(pid) && pid > 0)
  const taskkillFailures: TaskkillFailure[] = []

  for (const pid of pids) {
    try {
      deps.killTree(pid)
    } catch (error) {
      taskkillFailures.push({ pid, error: errorText(error) })
    }
  }

  const remainingPids = pids.filter(pid => deps.isPidAlive(pid))
  const alive = new Set(remainingPids)
  const clearedLocks: string[] = []
  const retainedLocks: string[] = []

  for (const lock of locks) {
    if (lockIsHeld(lock, pid => alive.has(pid) || deps.isPidAlive(pid))) {
      retainedLocks.push(lock.path)
      continue
    }

    deps.clearLock(lock.path)
    clearedLocks.push(lock.path)
  }

  const failedWhileAlive = new Set(taskkillFailures.map(failure => failure.pid))
  const liveFailure = remainingPids.length > 0 || pids.some(pid => failedWhileAlive.has(pid) && deps.isPidAlive(pid))

  return {
    taskkillFailures,
    remainingPids,
    clearedLocks,
    retainedLocks,
    liveFailure
  }
}

export function closeStopFailureMessage(result: CloseStopKillResult): string {
  const kills = result.taskkillFailures.map(failure => `PID ${failure.pid}: ${failure.error}`).join('; ')
  const remaining = result.remainingPids.length ? `still running: ${result.remainingPids.join(', ')}` : ''
  const retained = result.retainedLocks.length ? `locks still held: ${result.retainedLocks.join(', ')}` : ''

  return ['Windows close/stop did not finish cleanly.', kills, remaining, retained].filter(Boolean).join(' ')
}
