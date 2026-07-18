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
 * On terminal update failure the updater also writes
 * HERMES_HOME/.hermes-update-failed (`{failed_at_unix}`) and exits (#66356).
 * If a wedged updater somehow leaves the in-progress marker with a still-live
 * pid after failing, this module treats a failed sidecar at or after the
 * in-progress start time as "no live update" and prunes the stranded marker.
 *
 * This module holds the PURE, side-effect-light logic (path, pid liveness,
 * parse + staleness) so it is unit-testable without booting Electron. The
 * polling/boot-progress wrapper lives in main.ts where the boot-progress and
 * log sinks are.
 */

import fs from 'fs'
import path from 'path'

// Even with a live-looking PID, never treat a marker older than this as a live
// update. A full update (git pull + pip + desktop rebuild) is minutes, not tens
// of minutes; past this the marker is almost certainly stale (e.g. the OS
// recycled the pid onto an unrelated process), so the gate self-heals.
export const UPDATE_MARKER_MAX_AGE_MS = 20 * 60 * 1000

export function markerPath(hermesHome) {
  return path.join(hermesHome, '.hermes-update-in-progress')
}

export function failedMarkerPath(hermesHome) {
  return path.join(hermesHome, '.hermes-update-failed')
}

// True only if a host process with this pid is currently alive. Signal 0 does
// not deliver a signal — it just probes existence/permission. ESRCH => dead;
// EPERM => alive but owned by another user (still "alive" for our purposes).
// Injectable `kill` keeps it unit-testable.
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

function readFailedAtUnix(hermesHome) {
  try {
    const raw = fs.readFileSync(failedMarkerPath(hermesHome), 'utf8')
    const failedAt = Number.parseInt(String(raw).split('\n')[0].trim(), 10)

    return Number.isFinite(failedAt) ? failedAt : null
  } catch {
    return null
  }
}

function pruneMarkerFiles(hermesHome) {
  for (const file of [markerPath(hermesHome), failedMarkerPath(hermesHome)]) {
    try {
      fs.unlinkSync(file)
    } catch {
      void 0
    }
  }
}

/**
 * Read + interpret the marker.
 *
 * Returns `{ pid, ageMs }` only when an update is GENUINELY still running
 * (parseable pid that is alive, within the age ceiling, and not superseded by
 * a `.hermes-update-failed` sidecar). Returns `null` for every "no live
 * update" case — absent, unreadable, malformed, dead pid, past the ceiling,
 * or already-failed — and, when a stale marker file exists, deletes it so it
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
    maxAgeMs = UPDATE_MARKER_MAX_AGE_MS
  }: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
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
  const failedAt = readFailedAtUnix(hermesHome)
  // #66356: updater logged terminal failure but may still be alive with the
  // in-progress marker intact. A failed sidecar at/after startedAt wins.
  const alreadyFailed =
    failedAt != null && Number.isFinite(startedAt) && failedAt >= startedAt

  if (!alive || ageMs > maxAgeMs || alreadyFailed) {
    pruneMarkerFiles(hermesHome)

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
 * later overwrite it with its own PID — that's fine, the marker body is
 * the same format and `readLiveUpdateMarker` only cares that *some* live
 * pid owns it. When the updater finishes it deletes the marker as before.
 * If the updater never starts (spawn failure) the marker still contains a
 * real PID, so `readLiveUpdateMarker` will self-heal once that PID exits.
 *
 * Also clears any leftover `.hermes-update-failed` sidecar (mirrors
 * `run_update` on the installer side). A same-second leftover failed
 * timestamp would otherwise satisfy `failedAt >= startedAt` and prune the
 * brand-new in-progress marker (#66356 repair).
 */
export function writeUpdateMarker(hermesHome, pid, { now = Date.now } = {}) {
  const file = markerPath(hermesHome)
  const startedAt = Math.floor(now() / 1000)

  try {
    // Drop a prior-run failed sidecar before publishing the live marker so
    // `failedAt >= startedAt` cannot poison a fresh handoff.
    try {
      fs.unlinkSync(failedMarkerPath(hermesHome))
    } catch {
      void 0
    }
    fs.writeFileSync(file, `${pid}\n${startedAt}\n`, 'utf8')
  } catch {
    // Best-effort: if we can't write the marker, proceed anyway. The
    // updater will write its own when it reaches run_update.
  }
}
