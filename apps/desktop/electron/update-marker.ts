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

// ps flattens argv into a display string, losing boundaries for paths with
// spaces. Darwin's read-only KERN_PROCARGS2 preserves those boundaries.
const MAC_PROCESS_ARGV = `
import ctypes, json, struct, sys
lib = ctypes.CDLL(None)
mib = (ctypes.c_int * 3)(1, 49, int(sys.argv[1]))
size = ctypes.c_size_t()
if lib.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
    raise OSError("process arguments unavailable")
buf = ctypes.create_string_buffer(size.value)
if lib.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
    raise OSError("process arguments unavailable")
raw = buf.raw[:size.value]
argc = struct.unpack("i", raw[:4])[0]
start = raw.index(b"\\0", 4) + 1
while raw[start] == 0:
    start += 1
argv = [part.decode("utf-8") for part in raw[start:].split(b"\\0")[:argc]]
if argc < 1 or len(argv) != argc:
    raise ValueError("incomplete process arguments")
print(json.dumps(argv))
`

// Inline code is not an identity credential. Prove only the smallest harmless
// program; every other -c owner is unknown, including formal managed launchers.
const INLINE_PROCESS_IDENTITY = `
import ast, sys
try:
    body = ast.parse(sys.stdin.read()).body
    expr = body[0].value if len(body) == 1 and isinstance(body[0], ast.Expr) else None
    harmless = (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
        and expr.func.id == "print" and len(expr.args) > 0 and not expr.keywords
        and all(isinstance(arg, ast.Constant) and isinstance(arg.value, str) for arg in expr.args))
    print("non-update" if harmless else "unknown")
except Exception:
    print("unknown")
`

function isUpdateSubcommand(argv: string[], index: number): boolean {
  while (index < argv.length) {
    const argument = argv[index]

    if (argument === '--profile' || argument === '-p') {
      index += 2
    } else if (argument.startsWith('--profile=')) {
      index += 1
    } else {
      return argument === 'update'
    }
  }

  return false
}

function isUpdateCommand(argv: string[], inspectInline: (source: string) => boolean | null): boolean | null {
  const executable = path.basename(argv[0])

  if (executable === 'bash' || executable === 'sh') {
    // The stable owner execs /bin/bash with the script as argv[1], never -c.
    return Boolean(
      argv[1] &&
        path.isAbsolute(argv[1]) &&
        path.normalize(argv[1]).endsWith('/scripts/desktop-update/posix.sh')
    )
  }

  let entry = 0

  if (/^python(?:3(?:\.\d+)?)?$/i.test(executable)) {
    entry = 1

    while (['-I', '-S', '-E', '-s', '-u', '-B', '-O', '-OO'].includes(argv[entry])) {
      entry += 1
    }

    if (argv[entry] === '-c') {
      return argv[entry + 1] === undefined ? null : inspectInline(argv[entry + 1])
    }

    if (argv[entry] === '-m' && argv[entry + 1] === 'hermes_cli.main') {
      return isUpdateSubcommand(argv, entry + 2)
    }

    if (argv[entry]?.startsWith('-') && argv[entry] !== '-m') {
      return null
    }
  }

  if (path.basename(argv[entry] || '') !== 'hermes') {
    return false
  }

  return isUpdateSubcommand(argv, entry + 1)
}

/** Positive non-update identity clears a reused PID; unreadable argv retains the gate. */
export function isMacUpdateProcess(
  pid: number,
  inspectProcess: (command: string, args: string[], options: object) => string | Buffer = execFileSync
) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return true
  }

  try {
    const argv: unknown = JSON.parse(
      String(
        inspectProcess('/usr/bin/python3', ['-I', '-S', '-c', MAC_PROCESS_ARGV, String(pid)], {
          encoding: 'utf8',
          stdio: ['ignore', 'pipe', 'ignore'],
          timeout: 1500
        })
      )
    )

    if (!Array.isArray(argv) || argv.length === 0 || !argv[0] || !argv.every(arg => typeof arg === 'string')) {
      return true
    }

    return (
      isUpdateCommand(argv, source => {
        const verdict = String(
          inspectProcess('/usr/bin/python3', ['-I', '-S', '-c', INLINE_PROCESS_IDENTITY], {
            encoding: 'utf8',
            input: source,
            stdio: ['pipe', 'pipe', 'ignore'],
            timeout: 1500
          })
        ).trim()

        return verdict === 'non-update' ? false : null
      }) !== false
    )
  } catch {
    return true
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
    isExpectedUpdateProcess,
    processState = posixProcessState
  }: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
    /**
     * Optional platform-specific identity check for a live marker owner.
     * Return false only when the PID is positively known not to be an updater;
     * errors should return true so a healthy update is never interrupted.
     */
    isExpectedUpdateProcess?: (pid: number) => boolean
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

  const zombie = alive && isZombieState(processState(pid))
  let expected = true

  if (alive && !zombie && isExpectedUpdateProcess) {
    try {
      expected = isExpectedUpdateProcess(pid)
    } catch {
      // The identity probe is supplementary. Losing ps permission or a
      // transient inspection error must not interrupt an active update.
      expected = true
    }
  }

  if (!alive || zombie || !expected || ageMs > maxAgeMs) {
    try {
      fs.unlinkSync(file)
    } catch {
      void 0
    }

    return null
  }

  return { pid, ageMs }
}

/** Host-aware boot gate: only macOS needs the additional PID identity probe. */
export function hasLiveUpdateMarker(hermesHome) {
  return Boolean(
    readLiveUpdateMarker(hermesHome, {
      isExpectedUpdateProcess: process.platform === 'darwin' ? isMacUpdateProcess : undefined
    })
  )
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
 * holder's original timestamp is preserved across those transfers so retries
 * cannot keep resetting the 20-minute stale ceiling. When the updater finishes
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
    startedAt
  }: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
    startedAt?: number
  } = {}
) {
  const file = markerPath(hermesHome)
  const nowMs = now()
  const owner = readLiveUpdateMarker(hermesHome, { kill, maxAgeMs, now: () => nowMs })

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
