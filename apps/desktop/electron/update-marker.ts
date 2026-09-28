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
import { execFile, execFileSync } from 'node:child_process'
import path from 'path'

import { hiddenWindowsChildOptions } from './windows-child-options'

// Past this age a live PID is only honored once its identity is verified
// (markerOwnerIsLive). The ceiling exists because the OS can recycle a dead
// updater's pid onto an unrelated process — not because updates are short: a
// Windows update with a desktop rebuild takes 40+ minutes (#109795). Keep in
// sync with UPDATE_MARKER_MAX_AGE_SECONDS in hermes_cli/update_lock.py and
// UPDATE_MARKER_MAX_AGE_SECS in apps/bootstrap-installer/src-tauri/src/update.rs.
export const UPDATE_MARKER_MAX_AGE_MS = 20 * 60 * 1000

// The owner wrote the marker (or was spawned just before the desktop wrote it),
// so it was created no later than the marker file's last write; a pid created
// after that write is recycled. Slack for 1s `ps` resolution. Keep in sync with
// PID_REUSE_TOLERANCE_SECONDS in hermes_cli/update_lock.py.
export const PID_REUSE_TOLERANCE_MS = 5_000

/**
 * The liveness decision shared (by contract) with update_lock.py and the Rust
 * updater. `ownerStartedMs` is the owner's creation time: a number when known,
 * `null` when unverifiable, `undefined` while an async probe is still pending
 * (treated as live — never hand the tree to a second updater on a guess).
 */
export function markerOwnerIsLive({
  pidAlive,
  ageMs,
  maxAgeMs = UPDATE_MARKER_MAX_AGE_MS,
  ownerStartedMs,
  markerWrittenMs
}: {
  pidAlive: boolean
  ageMs: number
  maxAgeMs?: number
  ownerStartedMs: number | null | undefined
  markerWrittenMs: number | null
}): boolean {
  if (!pidAlive) {
    return false
  }

  if (ageMs <= maxAgeMs) {
    return true
  }

  if (ownerStartedMs === undefined) {
    return true
  }

  if (ownerStartedMs === null || markerWrittenMs === null) {
    return false
  }

  return ownerStartedMs <= markerWrittenMs + PID_REUSE_TOLERANCE_MS
}

/** Seconds from `ps -o etime=` (`[[dd-]hh:]mm:ss`, shared by macOS and procps). */
export function parsePsEtime(text: string): number | null {
  const trimmed = String(text || '').trim()
  const dash = trimmed.lastIndexOf('-')
  const days = dash >= 0 ? trimmed.slice(0, dash) : ''
  const parts = (dash >= 0 ? trimmed.slice(dash + 1) : trimmed).split(':')

  if (parts.length < 2 || parts.length > 3 || !parts.every(p => /^\d+$/.test(p)) || (days && !/^\d+$/.test(days))) {
    return null
  }

  const seconds = parts.reduce((acc, part) => acc * 60 + Number(part), 0)

  return seconds + Number(days || 0) * 86_400
}

/** Wall-clock creation time (epoch ms) of `pid`; rejects when unverifiable. */
export function probeProcessStartMs(pid: number): Promise<number> {
  const isWindows = process.platform === 'win32'

  const [command, args] = isWindows
    ? [
        'powershell.exe',
        [
          '-NoProfile',
          '-NonInteractive',
          '-Command',
          `[DateTimeOffset]::new((Get-Process -Id ${pid} -ErrorAction Stop).StartTime).ToUnixTimeMilliseconds()`
        ]
      ]
    : ['ps', ['-o', 'etime=', '-p', String(pid)]]

  return new Promise((resolve, reject) => {
    // PowerShell 5.1 cold starts take seconds (#87169); this runs off the
    // polling path, so give it headroom.
    const child = execFile(
      command,
      args,
      hiddenWindowsChildOptions({ encoding: 'utf8', timeout: 30_000 }),
      (error, stdout) => {
        if (error) {
          return reject(error)
        }

        const text = String(stdout || '').trim()

        if (isWindows) {
          return /^\d+$/.test(text) ? resolve(Number(text)) : reject(new Error(`bad start time for ${pid}`))
        }

        const elapsed = parsePsEtime(text)

        return elapsed === null ? reject(new Error(`bad etime for ${pid}`)) : resolve(Date.now() - elapsed * 1000)
      }
    )

    child.stdin?.end()
  })
}

// A probe result is trusted for this long before it is refreshed, so a pid
// that dies and is recycled while the marker stays put is re-examined.
const OWNER_START_CACHE_MS = 60_000
const ownerStartCache = new Map<number, { value: number | null | undefined; at: number }>()

/**
 * Sync view of an async probe so the 1s gate poll never blocks the main
 * process on PowerShell: `undefined` while the first probe is in flight, then
 * the cached result (refreshed in the background once it ages out).
 */
export function cachedProcessStartMs(
  pid: number,
  { now = Date.now, probe = probeProcessStartMs }: { now?: () => number; probe?: (pid: number) => Promise<number> } = {}
): number | null | undefined {
  const hit = ownerStartCache.get(pid)

  if (hit && (hit.value === undefined || now() - hit.at < OWNER_START_CACHE_MS)) {
    return hit.value
  }

  const entry = { value: hit?.value, at: now() }
  ownerStartCache.set(pid, entry)
  probe(pid).then(
    value => {
      entry.value = value
      entry.at = now()
    },
    () => {
      entry.value = null
      entry.at = now()
    }
  )

  return hit?.value
}

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
 * (parseable pid that is alive, within the age ceiling or with a verified
 * identity past it — see markerOwnerIsLive). Returns `null` for every "no
 * live update" case — absent, unreadable, malformed, dead pid, or a recycled
 * / unverifiable pid past the ceiling — and, when a stale marker file exists,
 * deletes it so it cannot strand future launches.
 *
 * Pure-ish: file I/O against the given path, plus an injectable pid probe,
 * owner start-time lookup and clock for tests.
 */
export function readLiveUpdateMarker(
  hermesHome,
  {
    kill,
    now = Date.now,
    maxAgeMs = UPDATE_MARKER_MAX_AGE_MS,
    processState = posixProcessState,
    ownerStartedMs = cachedProcessStartMs
  }: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
    /** Injectable override of the zombie/state probe (see posixProcessState). */
    processState?: (pid: number) => string | null
    ownerStartedMs?: (pid: number) => number | null | undefined
  } = {}
) {
  const file = markerPath(hermesHome)
  let raw
  let markerWrittenMs: number | null

  try {
    raw = fs.readFileSync(file, 'utf8')
    markerWrittenMs = fs.statSync(file).mtimeMs
  } catch {
    return null // absent or unreadable => no live update
  }

  const [pidLine, startedLine] = String(raw).split('\n')
  const pid = Number.parseInt((pidLine || '').trim(), 10)
  const startedAt = Number.parseInt((startedLine || '').trim(), 10)
  const ageMs = Number.isFinite(startedAt) ? now() - startedAt * 1000 : Infinity
  // A zombie answers signal 0 like a live process but is dead for every
  // liveness decision here (#77259).
  const alive = Number.isInteger(pid) && isPidAlive(pid, kill) && !isZombieState(processState(pid))
  // Only a well-formed marker past the ceiling pays for the identity probe.
  const probe = alive && ageMs > maxAgeMs && Number.isFinite(startedAt)

  if (
    !markerOwnerIsLive({
      pidAlive: alive,
      ageMs,
      maxAgeMs,
      ownerStartedMs: probe ? ownerStartedMs(pid) : null,
      markerWrittenMs
    })
  ) {
    try {
      fs.unlinkSync(file)
    } catch {
      void 0
    }

    return null
  }

  return { pid, ageMs }
}

/**
 * Write the update-in-progress marker *from the desktop* before handing off
 * to the detached updater.
 *
 * The Tauri-based hermes-setup.exe takes several seconds to initialise its
 * window and reach the Rust `run_update` entry point where it writes the
 * marker itself. During that gap the desktop's `app.quit()` teardown kills
 * the backend child, the renderer's WebSocket drops, and the renderer
 * immediately calls `ensureBackend()` → `waitForUpdateToFinish()`. Because
 * the updater hasn't written the marker yet, the gate sees no live update
 * and spawns a *new* backend — which re-locks `.pyd` files in the venv.
 * When the updater finally reaches the venv-rebuild stage it finds those
 * files locked and the update bricks.
 *
 * Fix: the desktop writes the marker itself, using the spawned updater's
 * PID, immediately after `spawn()`. The updater's `UpdateMarkerGuard` will
 * later adopt it or another hand-off stage may replace the PID. A live
 * holder's original timestamp is preserved across those transfers; the file's
 * mtime still moves with each write, which is what the pid-reuse check compares
 * against, so the new owner (spawned before this write) verifies as live past
 * the 20-minute ceiling. When the updater finishes
 * it deletes the marker as before.
 * If the updater never starts (spawn failure) the marker still contains a
 * real PID, so `readLiveUpdateMarker` will self-heal once that PID exits.
 */
export function writeUpdateMarker(
  hermesHome,
  pid,
  {
    kill,
    now = Date.now,
    maxAgeMs = UPDATE_MARKER_MAX_AGE_MS,
    ownerStartedMs,
    startedAt
  }: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
    ownerStartedMs?: (pid: number) => number | null | undefined
    startedAt?: number
  } = {}
) {
  const file = markerPath(hermesHome)
  const nowMs = now()
  const owner = readLiveUpdateMarker(hermesHome, { kill, maxAgeMs, now: () => nowMs, ownerStartedMs })

  const acquiredAt =
    typeof startedAt === 'number' && Number.isInteger(startedAt)
      ? startedAt
      : owner
        ? Math.floor((nowMs - owner.ageMs) / 1000)
        : Math.floor(nowMs / 1000)

  try {
    fs.writeFileSync(file, `${pid}\n${acquiredAt}\n`, 'utf8')
  } catch {
    // Best-effort: if we can't write the marker, proceed anyway. The
    // updater will write its own when it reaches run_update.
  }
}

/**
 * Whether a NEW updater hand-off must be refused because a different,
 * already-alive updater currently owns the marker (#75778).
 *
 * `writeUpdateMarker` unconditionally overwrites the marker file. Called
 * before every hand-off with no conflict check, a user who clicks "Update"
 * again while a prior updater is still parked mid-run (e.g. "waiting for
 * Hermes to exit…") clobbers that still-running updater's claim: the
 * retry's pre-write now names the NEW child, so the OLD process — alive
 * and mutating the checkout — is no longer recorded as the owner. A second
 * live updater can then run over the same tree unrecorded, the exact
 * two-updaters-at-once hazard `UpdateMarkerGuard` in the Rust updater
 * exists to prevent (apps/bootstrap-installer/src-tauri/src/update.rs).
 *
 * Returns the live foreign owner (with a ready-to-show message) when the
 * hand-off must be refused, or `null` when it's safe to spawn — no marker,
 * or the existing one is stale/dead and self-heals via
 * `readLiveUpdateMarker`.
 */
export function updateHandoffConflict(
  hermesHome,
  opts: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
    ownerStartedMs?: (pid: number) => number | null | undefined
  } = {}
) {
  const owner = readLiveUpdateMarker(hermesHome, opts)

  if (!owner) {
    return null
  }

  const mins = Math.floor(owner.ageMs / 60_000)
  const secs = Math.floor((owner.ageMs % 60_000) / 1000)
  const elapsed = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`

  return {
    pid: owner.pid,
    ageMs: owner.ageMs,
    message: `An update is already running (PID ${owner.pid}, started ${elapsed} ago). Wait for it to finish, then try again.`
  }
}
