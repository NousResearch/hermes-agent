/**
 * Recover an orphaned Chromium/Electron singleton lock on Linux.
 *
 * Electron's requestSingleInstanceLock() is supposed to let a new process
 * take over after the old process dies. On some Linux/X11 combinations a
 * dead SingletonLock symlink can still make the new process look like the
 * losing instance, which then exits before the window or desktop log exists.
 * Keep the preflight small, conservative, and dependency-free so it can run
 * before app.requestSingleInstanceLock() and be unit-tested without Electron.
 */

import fs, { type PathLike, type Stats } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { isPidAlive } from './update-marker'

const SINGLETON_NAMES = ['SingletonLock', 'SingletonCookie', 'SingletonSocket'] as const

export type SingletonLockResult = {
  pid: number
  target: string
  cleared: string[]
}

export type SingletonLockDeps = {
  hostname?: string
  isPidAlive?: (pid: number) => boolean
  lstatSync?: (path: PathLike) => Stats
  readlinkSync?: (path: PathLike) => string
  unlinkSync?: (path: PathLike) => void
}

/** Parse Chromium's local `hostname-pid` SingletonLock target. */
export function parseLocalSingletonLockTarget(target: string, hostname = os.hostname()) {
  const delimiter = target.lastIndexOf('-')

  if (delimiter <= 0 || target.slice(0, delimiter) !== hostname) {
    return null
  }

  const pid = Number(target.slice(delimiter + 1))

  if (!Number.isSafeInteger(pid) || pid <= 0) {
    return null
  }

  return { hostname, pid }
}

/**
 * Remove only symlink-based singleton entries belonging to a dead local PID.
 *
 * Regular files, foreign-host locks, malformed targets, live PIDs, and
 * unreadable paths are deliberately left untouched. The lock target is
 * re-read immediately before mutation so a normal startup race is unlikely
 * to remove a lock created by a concurrent live instance.
 */
export function clearStaleLinuxSingletonLock(
  userDataDir: string,
  {
    hostname = os.hostname(),
    isPidAlive: pidAlive = isPidAlive,
    lstatSync = fs.lstatSync as (path: PathLike) => Stats,
    readlinkSync = fs.readlinkSync as (path: PathLike) => string,
    unlinkSync = fs.unlinkSync as (path: PathLike) => void
  }: SingletonLockDeps = {}
): SingletonLockResult | null {
  const lockPath = path.join(userDataDir, 'SingletonLock')
  let lockStat

  try {
    lockStat = lstatSync(lockPath)
  } catch {
    return null
  }

  if (!lockStat.isSymbolicLink()) {
    return null
  }

  let target

  try {
    target = readlinkSync(lockPath)
  } catch {
    return null
  }

  const parsed = parseLocalSingletonLockTarget(target, hostname)

  if (!parsed || pidAlive(parsed.pid)) {
    return null
  }

  // Avoid acting on a lock that changed while the PID probe was running.
  try {
    if (readlinkSync(lockPath) !== target) {
      return null
    }
  } catch {
    return null
  }

  const cleared: string[] = []

  for (const name of SINGLETON_NAMES) {
    const entryPath = path.join(userDataDir, name)
    let entryStat

    try {
      entryStat = lstatSync(entryPath)
    } catch {
      continue
    }

    // Never unlink a regular file. Chromium's singleton entries are links on
    // the Linux path, and a regular file may be owned by an older/other
    // implementation with semantics we cannot safely infer.
    if (!entryStat.isSymbolicLink()) {
      continue
    }

    try {
      unlinkSync(entryPath)
      cleared.push(name)
    } catch {
      // Best-effort recovery must never block a launch.
    }
  }

  return { pid: parsed.pid, target, cleared }
}
