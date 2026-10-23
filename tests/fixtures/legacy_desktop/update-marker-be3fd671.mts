// The update-marker reader a packaged Desktop shipped at be3fd671d70 (apps/desktop/electron/update-marker.ts
// lines 1-160, verbatim): the old reader new hand-off scripts must keep parked. Never edit it.
/**
 * In-app update mutual-exclusion marker (#50238).
 *
 * The Tauri updater writes HERMES_HOME/.hermes-update-in-progress for the whole
 * duration of an `--update` run (see apps/bootstrap-installer/src-tauri/src/
 * update.rs `UpdateMarkerGuard`). The marker body is two lines: the updater's
 * pid and the unix-seconds it started.
 *
 * Why: if the user relaunches the desktop mid-update — the window vanished with
 * no progress and looks crashed — a fresh instance must NOT spawn its own local
 * backend. That backend re-locks the venv shim, the updater's straggler cleanup
 * (`force_kill_other_hermes`, taskkill /IM hermes.exe) kills it, the launch
 * fails with the 45s "backend didn't come up" timeout, and the user relaunches
 * into the same trap — an infinite respawn/kill loop. The desktop gates local
 * backend startup on this marker and parks until the update finishes.
 *
 * This module holds the PURE, side-effect-light logic (path, pid liveness,
 * parse + staleness) so it is unit-testable without booting Electron. The
 * polling/boot-progress wrapper lives in main.ts where the boot-progress and
 * log sinks are.
 */

import fs from 'fs'
import { execFileSync } from 'node:child_process'
import path from 'path'

// Even with a live-looking PID, never treat a marker older than this as a live
// update. A full update (git pull + pip + desktop rebuild) is minutes, not tens
// of minutes; past this the marker is almost certainly stale (e.g. the OS
// recycled the pid onto an unrelated process), so the gate self-heals.
export const UPDATE_MARKER_MAX_AGE_MS = 20 * 60 * 1000

export function markerPath(hermesHome) {
  return path.join(hermesHome, '.hermes-update-in-progress')
}

// True only if a host process with this pid is currently alive. Signal 0 does
// not deliver a signal — it just probes existence/permission. ESRCH => dead;
// EPERM => alive but owned by another user (still "alive" for our purposes).
// Injectable `kill` keeps it unit-testable.
//
// NOT zombie-aware on its own: signal 0 also succeeds for a process that
// exited but whose parent has not reaped it. Callers deciding whether an
// update marker's owner is still running must layer `posixProcessState` on
// top (see `readLiveUpdateMarker`).
export function isPidAlive(pid, kill: typeof process.kill = process.kill.bind(process)) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false
  }

  try {
    kill(pid, 0)

    return true
  } catch (err) {
    return Boolean(err && err.code === 'EPERM')
  }
}

/**
 * Single-letter process state (`ps` style) for a kill(0)-alive pid, or null
 * when it cannot be determined.
 *
 * A ZOMBIE — exited, still in the table because its parent has not reaped
 * it — answers signal 0 like a live process. A crashed updater lingering
 * that way would keep its update marker "live" and park the desktop boot
 * gate for the whole 20-minute ceiling (#77259, #120635, #125932). Linux
 * exposes the state via /proc; on macOS `ps -o stat=` does. Failures return
 * null so callers keep their signal-0 verdict (fail-open to alive, matching
 * the EPERM behavior above).
 */
export function posixProcessState(pid: number): string | null {
  if (process.platform === 'linux') {
    try {
      const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8')
      const commEnd = stat.lastIndexOf(')')
      const state = commEnd >= 0 ? stat.slice(commEnd + 2, commEnd + 3) : ''

      return state || null
    } catch {
      return null
    }
  }

  if (process.platform === 'darwin') {
    try {
      const out = execFileSync('ps', ['-o', 'stat=', '-p', String(pid)], {
        encoding: 'utf8',
        timeout: 5000
      })

      return out.trim().charAt(0) || null
    } catch {
      return null
    }
  }

  return null
}

// A state of 'Z'/'Z+' (and friends) means the process exited and only its
// unreaped table entry remains — dead for every liveness decision here.
function isZombieState(state: string | null | undefined): boolean {
  return Boolean(state && state.toUpperCase().startsWith('Z'))
}

/**
 * Read + interpret the marker.
 *
 * Returns `{ pid, ageMs }` only when an update is GENUINELY still running
 * (parseable pid that is alive, within the age ceiling). Returns `null` for
 * every "no live update" case — absent, unreadable, malformed, dead pid, or
 * past the ceiling — and, when a stale marker file exists, deletes it so it
 * cannot strand future launches.
 *
 * Pure-ish: file I/O against the given path, plus an injectable pid probe and
 * clock for tests.
 */
export function readLiveUpdateMarker(
  hermesHome,
  {
    kill,
    now = Date.now,
    maxAgeMs = UPDATE_MARKER_MAX_AGE_MS,
    processState = posixProcessState
  }: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
    /** Injectable override of the zombie/state probe (see posixProcessState). */
    processState?: (pid: number) => string | null
  } = {}
) {
  const file = markerPath(hermesHome)
  let raw

  try {
    raw = fs.readFileSync(file, 'utf8')
  } catch {
    return null // absent or unreadable => no live update
  }

  const [pidLine, startedLine] = String(raw).split('\n')
  const pid = Number.parseInt((pidLine || '').trim(), 10)
  const startedAt = Number.parseInt((startedLine || '').trim(), 10)
  const ageMs = Number.isFinite(startedAt) ? now() - startedAt * 1000 : Infinity
  const alive = Number.isInteger(pid) && isPidAlive(pid, kill)

  if (!alive || isZombieState(processState(pid)) || ageMs > maxAgeMs) {
    try {
      fs.unlinkSync(file)
    } catch {
      void 0
    }

    return null
  }

  return { pid, ageMs }
}
