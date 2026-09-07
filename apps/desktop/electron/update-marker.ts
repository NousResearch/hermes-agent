/**
 * In-app update mutual-exclusion marker (#50238).
 *
 * The Tauri updater writes HERMES_HOME/.hermes-update-in-progress for the whole
 * duration of an `--update` run (see apps/bootstrap-installer/src-tauri/src/
 * update.rs `UpdateMarkerGuard`). The marker body starts with the updater's
 * pid and the unix-seconds it started. A Desktop-owned Windows bridge adds a
 * third `handoff-bridge` line until windows.ps1 claims the marker itself.
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
import { randomUUID } from 'node:crypto'
import path from 'path'

import { getCachedWindowsProcessCreatedAt } from './windows-process-identity'

export { transferUpdateMarkerIfOwnedBy } from './windows-update-marker'

// A full update (git pull + pip + desktop rebuild) can be slow, so an owner
// PROVEN to be the claimant (`matching`) remains authoritative past this age
// and a second updater can never race it into the same install. An owner whose
// creation time is unprovable (`unknown` — a live pid that may just be a
// recycled number) expires here, as upstream main does today; without that a
// single recycled pid wedges every future update and the MCP bridge forever.
export const UPDATE_MARKER_MAX_AGE_MS = 20 * 60 * 1000
export const UPDATE_MARKER_CLOCK_SKEW_MS = 5 * 1000

// A marker that is empty, malformed or future-dated is not a claim — nothing
// alive is provably behind it. It is still not reclaimed on sight: windows.ps1
// and the Rust updater rewrite the marker in place, so a reader can catch a
// mid-write body. Observe the exact same bytes across this dwell first, then
// reclaim with an exact-content CAS, so a real write in flight always wins.
export const UPDATE_MARKER_DWELL_MS = 2 * 1000

export type ProcessCreateTimeProbe = (pid: number) => number | null | undefined
export type PidIdentityStatus = 'matching' | 'stale' | 'unknown'
export type UpdateMarkerBlockedReason = 'unreadable' | 'malformed' | 'future' | 'cleanup-race'

export interface UpdateMarkerClaim {
  pid: number
  startedAt: number
}

export interface LiveUpdateMarker extends UpdateMarkerClaim {
  kind: 'live'
  ageMs: number
  overdue: boolean
  bridge?: boolean
}

export interface UnreadableUpdateMarker {
  kind: 'unreadable'
  reason: UpdateMarkerBlockedReason
  /** The exact file a human has to look at; every message must name it. */
  file: string
  message: string
  pid: null
  ageMs: null
  overdue: null
}

export type UpdateMarkerState = LiveUpdateMarker | UnreadableUpdateMarker

interface PidIdentityProbeOptions {
  kill?: typeof process.kill
  getProcessCreatedAt?: ProcessCreateTimeProbe
}

interface UpdateMarkerReadOptions extends PidIdentityProbeOptions {
  now?: () => number
  maxAgeMs?: number
  dwellMs?: number
}

// The Windows script starts through a short-lived cmd.exe wrapper. Hold the
// backend gate closed across that bounded wrapper-to-script claim gap, even
// after the bridge owner exits. The bound applies regardless of PID liveness
// so a failed hand-off cannot wedge retries behind the still-running Desktop.
export const UPDATE_HANDOFF_BRIDGE_GRACE_MS = 30 * 1000

export function markerPath(hermesHome) {
  return path.join(hermesHome, '.hermes-update-in-progress')
}

// True only if a host process with this pid is currently alive. Signal 0 does
// not deliver a signal — it just probes existence/permission. ESRCH => dead;
// EPERM => alive but owned by another user (still "alive" for our purposes).
// Injectable `kill` keeps it unit-testable.
export function isPidAlive(pid, kill: typeof process.kill = process.kill.bind(process)) {
  if (!Number.isSafeInteger(pid) || pid <= 0) {
    return false
  }

  try {
    kill(pid, 0)

    return true
  } catch (err: any) {
    return err?.code !== 'ESRCH'
  }
}

export function probePidIdentity(
  pid: number,
  startedAtSeconds: number,
  { kill = process.kill.bind(process), getProcessCreatedAt }: PidIdentityProbeOptions = {}
): PidIdentityStatus {
  if (!Number.isSafeInteger(pid) || pid <= 0 || !Number.isSafeInteger(startedAtSeconds) || startedAtSeconds <= 0) {
    return 'stale'
  }

  try {
    kill(pid, 0)
  } catch (error: any) {
    if (error?.code === 'ESRCH') {
      return 'stale'
    }
  }

  if (!getProcessCreatedAt) {
    return 'unknown'
  }

  let createdAt: number | null | undefined

  try {
    createdAt = getProcessCreatedAt(pid)
  } catch {
    return 'unknown'
  }

  if (!Number.isFinite(createdAt) || Number(createdAt) <= 0) {
    return 'unknown'
  }

  return Number(createdAt) < startedAtSeconds + 1 ? 'matching' : 'stale'
}

export function pidMatchesUpdateOwner(
  pid: number,
  startedAtSeconds: number,
  options: PidIdentityProbeOptions = {}
): boolean {
  return probePidIdentity(pid, startedAtSeconds, options) !== 'stale'
}

function parsePositiveInteger(line: string): number | null {
  const normalized = line.trim()

  if (!/^[0-9]+$/.test(normalized)) {
    return null
  }

  const value = Number(normalized)

  return Number.isSafeInteger(value) && value > 0 ? value : null
}

function parseMarker(raw: string): { pid: number; startedAt: number; bridge: boolean } | null {
  const lines = raw.split(/\r?\n/)

  if (lines.at(-1) === '') {
    lines.pop()
  }

  if (lines.length !== 2 && lines.length !== 3) {
    return null
  }

  const pid = parsePositiveInteger(lines[0])
  const startedAt = parsePositiveInteger(lines[1])

  if (pid === null || startedAt === null) {
    return null
  }

  if (lines.length === 3 && lines[2] !== 'handoff-bridge') {
    return null
  }

  return { pid, startedAt, bridge: lines.length === 3 }
}

type RestoreResult = 'restored' | 'superseded' | 'unresolved'

function restoreIsolatedMarker(file: string, tombstone: string): RestoreResult {
  try {
    fs.linkSync(tombstone, file)
    fs.unlinkSync(tombstone)

    return 'restored'
  } catch (error: any) {
    if (error?.code === 'EEXIST') {
      try { fs.unlinkSync(tombstone) } catch { void 0 }

      return 'superseded'
    }
  }

  let payload: Buffer

  try {
    payload = fs.readFileSync(tombstone)
  } catch {
    return 'unresolved'
  }

  let descriptor: number | null = null

  try {
    descriptor = fs.openSync(file, 'wx', 0o600)
    fs.writeFileSync(descriptor, payload)
    fs.fsyncSync(descriptor)
    fs.closeSync(descriptor)
    descriptor = null

    try { fs.unlinkSync(tombstone) } catch { void 0 }

    return 'restored'
  } catch (error: any) {
    if (descriptor !== null) {
      try { fs.closeSync(descriptor) } catch { void 0 }
    }

    if (error?.code === 'EEXIST') {
      try { fs.unlinkSync(tombstone) } catch { void 0 }

      return 'superseded'
    }

    return 'unresolved'
  }
}

function recoveryArtifacts(file: string): string[] | null {
  try {
    return fs.readdirSync(path.dirname(file))
      .filter(name => name.startsWith(`${path.basename(file)}.cas-`))
      .map(name => path.join(path.dirname(file), name))
  } catch (error: any) {
    return error?.code === 'ENOENT' ? [] : null
  }
}

function removeMarkerIfExact(file: string, expectedRaw: string): 'retry' | 'unresolved' {
  const tombstone = `${file}.cas-release-${process.pid}-${randomUUID()}`

  try { fs.renameSync(file, tombstone) } catch { return 'retry' }

  let isolated: string | null = null

  try { isolated = fs.readFileSync(tombstone, 'utf8') } catch { void 0 }

  if (isolated !== expectedRaw) {
    return restoreIsolatedMarker(file, tombstone) === 'unresolved' ? 'unresolved' : 'retry'
  }

  try {
    fs.unlinkSync(tombstone)

    return 'retry'
  } catch {
    return restoreIsolatedMarker(file, tombstone) === 'unresolved' ? 'unresolved' : 'retry'
  }
}

const BLOCKED_MARKER_CAUSE: Record<UpdateMarkerBlockedReason, string> = {
  unreadable: 'the update marker could not be read',
  malformed: 'the update marker is empty or malformed',
  future: 'the update marker is dated in the future',
  'cleanup-race': 'an update marker cleanup is still unresolved'
}

/**
 * Never surface a blocked marker without the path. Recovery is deleting one
 * file; users spent whole sessions on "Hermes is still updating" because no
 * message ever said which file, and the self-heal below can be defeated by a
 * clock jump or a read-only directory.
 */
export function blockedUpdateMarkerMessage(file: string, reason: UpdateMarkerBlockedReason): string {
  return (
    `Hermes cannot verify the update marker: ${BLOCKED_MARKER_CAUSE[reason]}. ` +
    `If no update is running, delete this file and retry: ${file}`
  )
}

function blockedMarker(file: string, reason: UpdateMarkerBlockedReason): UnreadableUpdateMarker {
  return {
    kind: 'unreadable',
    reason,
    file,
    message: blockedUpdateMarkerMessage(file, reason),
    pid: null,
    ageMs: null,
    overdue: null
  }
}

// First sighting of an unclaimable body, keyed by marker path. Deliberately
// process-local and content-keyed: a marker that changes between polls restarts
// the dwell, so an in-flight rewrite is never reclaimed out from under its
// writer. Only ever grows by one entry per HERMES_HOME in a process.
const dwellSightings = new Map<string, { raw: string; firstSeenAt: number }>()

/**
 * Whether an unclaimable marker body has now sat unchanged long enough to
 * reclaim. Returns false (and records/refreshes the sighting) otherwise.
 */
function unclaimableBodyHasSettled(file: string, raw: string, at: number, dwellMs: number): boolean {
  const seen = dwellSightings.get(file)

  if (!seen || seen.raw !== raw || !Number.isFinite(seen.firstSeenAt) || seen.firstSeenAt > at) {
    dwellSightings.set(file, { raw, firstSeenAt: at })

    return false
  }

  return at - seen.firstSeenAt >= dwellMs
}

/**
 * Read + interpret the marker.
 *
 * Returns `{ pid, ageMs }` when an update is still running: either a parseable
 * live pid, or an explicitly tagged Windows hand-off bridge inside its short
 * claim grace. A live owner remains authoritative after the advisory age
 * ceiling; only an authenticated stale owner is removed, so a slow update
 * cannot race a new updater into the same install.
 *
 * Pure-ish: file I/O against the given path, plus an injectable pid probe and
 * clock for tests.
 */
export function readLiveUpdateMarker(
  hermesHome,
  {
    kill,
    getProcessCreatedAt = kill ? undefined : getCachedWindowsProcessCreatedAt,
    now = Date.now,
    maxAgeMs = UPDATE_MARKER_MAX_AGE_MS,
    dwellMs = UPDATE_MARKER_DWELL_MS
  }: UpdateMarkerReadOptions = {}
): UpdateMarkerState | null {
  const file = markerPath(hermesHome)

  for (let attempt = 0; attempt < 3; attempt += 1) {
    let raw: string

    try {
      raw = fs.readFileSync(file, 'utf8')
    } catch (error: any) {
      if (error?.code === 'ENOENT') {
        dwellSightings.delete(file)

        const artifacts = recoveryArtifacts(file)

        if (artifacts === null) {
          return blockedMarker(file, 'cleanup-race')
        }

        if (artifacts.length === 0) {
          return null
        }

        const releasePrefix = `${path.basename(file)}.cas-release-`

        if (artifacts.length === 1 && path.basename(artifacts[0]).startsWith(releasePrefix)) {
          const releaserPid = Number(path.basename(artifacts[0]).slice(releasePrefix.length).split('-')[0])

          // A live releaser still owns the rename/read/unlink transaction.
          // Restoring its temporary file here would resurrect the very claim
          // it is removing. Only recover an abandoned release.
          if (!Number.isSafeInteger(releaserPid) || releaserPid <= 0 || isPidAlive(releaserPid, kill)) {
            return blockedMarker(file, 'cleanup-race')
          }

          if (restoreIsolatedMarker(file, artifacts[0]) === 'unresolved') {
            return blockedMarker(file, 'cleanup-race')
          }

          continue
        }

        return blockedMarker(file, 'cleanup-race')
      }

      return blockedMarker(file, 'unreadable')
    }

    const artifacts = recoveryArtifacts(file)

    if (artifacts === null || artifacts.length > 0) {
      return blockedMarker(file, 'cleanup-race')
    }

    // An empty/malformed/future body names nobody, so no owner can be harmed
    // by reclaiming it — but a torn write is momentarily indistinguishable
    // from one, so require the exact same bytes across the dwell first.
    const reclaimUnclaimable = (reason: UpdateMarkerBlockedReason): UpdateMarkerState | 'retry' => {
      if (!unclaimableBodyHasSettled(file, String(raw), now(), dwellMs)) {
        return blockedMarker(file, reason)
      }

      if (removeMarkerIfExact(file, String(raw)) === 'unresolved') {
        return blockedMarker(file, 'cleanup-race')
      }

      dwellSightings.delete(file)

      return 'retry'
    }

    const parsed = parseMarker(String(raw))

    if (!parsed) {
      const outcome = reclaimUnclaimable('malformed')

      if (outcome !== 'retry') { return outcome }

      continue
    }

    const { pid, startedAt, bridge } = parsed
    const ageMs = now() - startedAt * 1000

    if (!Number.isFinite(ageMs) || ageMs < -UPDATE_MARKER_CLOCK_SKEW_MS) {
      const outcome = reclaimUnclaimable('future')

      if (outcome !== 'retry') { return outcome }

      continue
    }

    dwellSightings.delete(file)

    if (bridge && ageMs > UPDATE_HANDOFF_BRIDGE_GRACE_MS) {
      if (removeMarkerIfExact(file, String(raw)) === 'unresolved') {
        return blockedMarker(file, 'cleanup-race')
      }

      continue
    }

    const identity = probePidIdentity(pid, startedAt, { kill, getProcessCreatedAt })
    const overdue = ageMs > maxAgeMs

    // `stale` is a proven-dead or proven-recycled pid. `unknown` is a live pid
    // we could not tie to this claim; it holds the gate until the age ceiling
    // and is then reclaimed, which is what keeps a marker recoverable at all.
    if (!bridge && (identity === 'stale' || (identity === 'unknown' && overdue))) {
      if (removeMarkerIfExact(file, String(raw)) === 'unresolved') {
        return blockedMarker(file, 'cleanup-race')
      }

      continue
    }

    return { kind: 'live', pid, startedAt, ageMs, overdue, ...(bridge ? { bridge: true } : {}) }
  }

  return blockedMarker(file, 'cleanup-race')
}

export type UpdateMarkerClaimResult =
  | { acquired: true; owner: LiveUpdateMarker }
  | { acquired: false; message: string }

/** Claim the shared marker before stopping holders or launching an updater. */
export function acquireUpdateMarker(
  hermesHome: string,
  { pid = process.pid, now = Date.now, ...probe }: UpdateMarkerReadOptions & { pid?: number } = {}
): UpdateMarkerClaimResult {
  const conflict = updateHandoffConflict(hermesHome, { ...probe, now })

  if (conflict) { return { acquired: false, message: conflict.message } }

  const startedAt = Math.floor(now() / 1000)

  if (!Number.isSafeInteger(pid) || pid <= 0 || !Number.isSafeInteger(startedAt) || startedAt <= 0) {
    return { acquired: false, message: 'Cannot establish the update marker owner identity.' }
  }

  const file = markerPath(hermesHome)
  const raw = `${pid}\n${startedAt}\n`
  let descriptor: number | undefined

  try {
    fs.mkdirSync(hermesHome, { recursive: true })
    descriptor = fs.openSync(file, 'wx', 0o600)
    fs.writeFileSync(descriptor, raw)
    fs.fsyncSync(descriptor)
  } catch {
    return {
      acquired: false,
      message: `Could not claim the update marker. Another update may have started; retry when it finishes. Marker: ${file}`
    }
  } finally {
    if (descriptor !== undefined) { fs.closeSync(descriptor) }
  }

  const artifacts = recoveryArtifacts(file)

  if (artifacts === null || artifacts.length > 0) {
    removeMarkerIfExact(file, raw)

    return { acquired: false, message: blockedUpdateMarkerMessage(file, 'cleanup-race') }
  }

  return { acquired: true, owner: { kind: 'live', pid, startedAt, ageMs: 0, overdue: false } }
}

/**
 * Write the update-in-progress marker *from the desktop* before handing off
 * to the detached updater.
 *
 * During updater startup the Desktop's backend exits and its renderer may
 * reconnect. Without a marker, that reconnect can spawn a new backend which
 * re-locks the venv before the updater reaches its rebuild stage.
 *
 * Staged updaters receive a marker with their spawned PID. The repo-owned
 * Windows script instead receives a tagged marker with the Desktop PID before
 * `cmd start`; the tag keeps the gate closed for the bounded claim gap, then
 * windows.ps1 replaces it with its own PID and creation time. New ownership
 * claims use acquireUpdateMarker; this compatibility writer is only for an
 * already-coordinated transfer. A transferred timestamp must still bound the
 * new owner's creation time.
 */
export function writeUpdateMarker(
  hermesHome,
  pid,
  {
    kill,
    now = Date.now,
    maxAgeMs = UPDATE_MARKER_MAX_AGE_MS,
    startedAt,
    handoffBridge = false
  }: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
    startedAt?: number
    handoffBridge?: boolean
  } = {}
) {
  const file = markerPath(hermesHome)
  const nowMs = now()
  const owner = readLiveUpdateMarker(hermesHome, { kill, maxAgeMs, now: () => nowMs })
  const liveOwner = owner?.kind === 'live' ? owner : null

  const acquiredAt =
    typeof startedAt === 'number' && Number.isInteger(startedAt)
      ? startedAt
      : liveOwner
        ? Math.floor((nowMs - liveOwner.ageMs) / 1000)
        : Math.floor(nowMs / 1000)

  try {
    fs.writeFileSync(file, `${pid}\n${acquiredAt}\n${handoffBridge ? 'handoff-bridge\n' : ''}`, 'utf8')
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
  opts: UpdateMarkerReadOptions = {}
) {
  const owner = readLiveUpdateMarker(hermesHome, opts)

  if (!owner) {
    return null
  }

  if (owner.kind === 'unreadable') {
    return { pid: null, ageMs: null, message: owner.message }
  }

  const mins = Math.floor(owner.ageMs / 60_000)
  const secs = Math.floor((owner.ageMs % 60_000) / 1000)
  const elapsed = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`

  return {
    pid: owner.pid,
    ageMs: owner.ageMs,
    message:
      `An update is already running (PID ${owner.pid}, started ${elapsed} ago). ` +
      `Wait for it to finish, then try again. Marker: ${markerPath(hermesHome)}`
  }
}

/**
 * Delete the marker only while it still names `pid` as its owner.
 *
 * The Desktop holds the marker under its own pid for the length of the
 * holder drain: the gateway and serve watchdogs in HERMES_HOME\gateway-service
 * treat `.hermes-update-in-progress` as "do not relaunch", and without it the
 * gateway watchdog brought the gateway back 30 s after `gateway stop --all`
 * on 2026-09-05, inside the lock gate. The hand-off script claims the marker
 * with CreateNew, so the Desktop must hand the slot back first; every refused
 * outcome hands it back as well.
 */
export function releaseUpdateMarkerIfOwnedBy(hermesHome: string, pid: number, expectedStartedAt?: number): boolean {
  const file = markerPath(hermesHome)
  let raw: string

  try {
    raw = fs.readFileSync(file, 'utf8')
  } catch {
    return false
  }

  const owner = parseMarker(raw)

  if (!owner || owner.pid !== pid || (expectedStartedAt !== undefined && owner.startedAt !== expectedStartedAt)) { return false }

  return removeMarkerIfExact(file, raw) === 'retry' && !fs.existsSync(file)
}
