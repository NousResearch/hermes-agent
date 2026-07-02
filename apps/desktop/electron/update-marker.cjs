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
 * polling/boot-progress wrapper lives in main.cjs where the boot-progress and
 * log sinks are.
 */

const fs = require('fs')
const path = require('path')

// Even with a live-looking PID, never treat a marker older than this as a live
// update. A full update (git pull + pip + desktop rebuild) is minutes, not tens
// of minutes; past this the marker is almost certainly stale (e.g. the OS
// recycled the pid onto an unrelated process), so the gate self-heals.
const UPDATE_MARKER_MAX_AGE_MS = 20 * 60 * 1000

function markerPath(hermesHome) {
  return path.join(hermesHome, '.hermes-update-in-progress')
}

// True only if a host process with this pid is currently alive. Signal 0 does
// not deliver a signal — it just probes existence/permission. ESRCH => dead;
// EPERM => alive but owned by another user (still "alive" for our purposes).
// Injectable `kill` keeps it unit-testable.
function isPidAlive(pid, kill = process.kill.bind(process)) {
  if (!Number.isInteger(pid) || pid <= 0) return false
  try {
    kill(pid, 0)
    return true
  } catch (err) {
    return Boolean(err && err.code === 'EPERM')
  }
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
function readLiveUpdateMarker(hermesHome, { kill, now = Date.now, maxAgeMs = UPDATE_MARKER_MAX_AGE_MS } = {}) {
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

  if (!alive || ageMs > maxAgeMs) {
    try {
      fs.unlinkSync(file)
    } catch {
      void 0
    }
    return null
  }
  return { pid, ageMs }
}

module.exports = {
  UPDATE_MARKER_MAX_AGE_MS,
  markerPath,
  isPidAlive,
  readLiveUpdateMarker
}
