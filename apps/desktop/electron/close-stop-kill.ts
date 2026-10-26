/**
 * Windows Desktop close/stop after the existing tree-kill.
 *
 * taskkill /T /F is not widened: the caller passes the owned PIDs and the
 * same tree-kill. This module does not discover extra processes to signal.
 * A taskkill error is recorded, never discarded. After the kill, owned PIDs
 * are inventoried, and only locks with no live holder are cleared.
 */

import fs from 'node:fs'
import path from 'node:path'

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
  /** Held locks, plus unheld ones whose removal failed (see lockErrors). */
  retainedLocks: string[]
  lockErrors: { path: string; error: string }[]
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
  const lockErrors: { path: string; error: string }[] = []

  for (const lock of locks) {
    if (lockIsHeld(lock, pid => alive.has(pid) || deps.isPidAlive(pid))) {
      retainedLocks.push(lock.path)

      continue
    }

    // An open handle (msvcrt byte lock) makes the delete fail: keep the lock
    // and report it rather than aborting the rest of close/stop.
    try {
      deps.clearLock(lock.path)
      clearedLocks.push(lock.path)
    } catch (error) {
      retainedLocks.push(lock.path)
      lockErrors.push({ path: lock.path, error: errorText(error) })
    }
  }

  const failedWhileAlive = new Set(taskkillFailures.map(failure => failure.pid))
  const liveFailure = remainingPids.length > 0 || pids.some(pid => failedWhileAlive.has(pid) && deps.isPidAlive(pid))

  return {
    taskkillFailures,
    remainingPids,
    clearedLocks,
    retainedLocks,
    lockErrors,
    liveFailure
  }
}

export function closeStopFailureMessage(result: CloseStopKillResult): string {
  const kills = result.taskkillFailures.map(failure => `PID ${failure.pid}: ${failure.error}`).join('; ')
  const remaining = result.remainingPids.length ? `still running: ${result.remainingPids.join(', ')}` : ''
  const retained = result.retainedLocks.length ? `locks still held: ${result.retainedLocks.join(', ')}` : ''

  return ['Windows close/stop did not finish cleanly.', kills, remaining, retained].filter(Boolean).join(' ')
}

export function holderPidsFromLockFile(lockPath: string): Pick<RuntimeLock, 'holderPids' | 'held'> {
  try {
    const raw = fs.readFileSync(lockPath, 'utf8').trim()

    if (!raw) {
      return { holderPids: [] }
    }

    let pid = Number(raw)

    if (!Number.isInteger(pid)) {
      const parsed = JSON.parse(raw) as { pid?: unknown }
      pid = Number(typeof parsed === 'object' && parsed ? parsed.pid : parsed)
    }

    return Number.isInteger(pid) && pid > 0 ? { holderPids: [pid] } : { holderPids: [] }
  } catch (error) {
    const code = (error as NodeJS.ErrnoException | null)?.code

    // Can't read the file: a live holder may be why. Do not clear it.
    if (code === 'EBUSY' || code === 'EPERM' || code === 'EACCES') {
      return { holderPids: [], held: true }
    }

    return { holderPids: [] }
  }
}

export function collectCloseStopLocks(hermesHome: string): RuntimeLock[] {
  const roots = [hermesHome]
  const profilesRoot = path.join(hermesHome, 'profiles')

  try {
    for (const name of fs.readdirSync(profilesRoot)) {
      roots.push(path.join(profilesRoot, name))
    }
  } catch {
    // No profiles directory — the default home lock is enough.
  }

  const locks: RuntimeLock[] = []

  for (const root of roots) {
    const lockPath = path.join(root, 'gateway.lock')

    try {
      if (!fs.statSync(lockPath).isFile()) {
        continue
      }
    } catch {
      continue
    }

    locks.push({ path: lockPath, ...holderPidsFromLockFile(lockPath) })
  }

  return locks
}
