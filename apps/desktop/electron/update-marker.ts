/**
 * The update marker `HERMES_HOME/.hermes-update-in-progress` (#50238), format v2.
 *
 * One file, shared by every updater (Python `hermes_cli/update_lock.py`, the Rust
 * `UpdateMarkerGuard`, `scripts/desktop-update/{windows.ps1,posix.sh}`) and this
 * Desktop. While it names a LIVE owner, a Desktop reopened mid-update must not
 * spawn a backend on the runtime being replaced, and no second update may start.
 *
 * Body (UTF-8, `\n` line ends):
 *
 *     <pid>
 *     <started_at unix seconds>
 *     ct:<owner creation time, unix seconds, 3 decimals>     (absent in v1 writers)
 *     delegate:<pid> ct:<creation time>                      (optional, see below)
 *
 * Liveness: the owner is LIVE iff its pid is alive (not a zombie) and its
 * creation time matches the recorded one within 2 s (an unprobeable creation
 * time counts as live). A LIVE v2 owner stays live regardless of age: a Windows
 * update is routinely 25-40 minutes. Only a v1 marker (no `ct:` line) keeps the
 * legacy 20-minute ceiling, since without a creation time it is the only thing
 * that tells a reused pid from the original owner. The marker is LIVE when the
 * owner OR the delegate (`hermes update` running under a hand-off script's
 * claim) is live, so a killed script cannot hide a still-mutating update.
 *
 * Deletes are compare-and-delete: re-read the bytes and unlink only when they
 * still equal the bytes judged dead (or the bytes we wrote). A verdict from an
 * earlier read never unlinks a newer claim.
 */

import fs from 'fs'
import { execFile, execFileSync } from 'node:child_process'
import path from 'path'

/** Legacy ceiling, applied ONLY to v1 markers (no creation time recorded). */
export const UPDATE_MARKER_MAX_AGE_MS = 20 * 60 * 1000

/** |recorded - actual| creation-time slack, in seconds (C1 rule 3). */
export const CREATE_TIME_TOLERANCE_S = 2.0

/** How long a hand-off script has to take the bridge marker (C2). */
export const HANDOFF_CLAIM_TIMEOUT_MS = 20_000

export function markerPath(hermesHome: string) {
  return path.join(hermesHome, '.hermes-update-in-progress')
}

// True only if a host process with this pid is currently alive. Signal 0 does
// not deliver a signal — it just probes existence/permission. ESRCH => dead;
// EPERM => alive but owned by another user (still "alive" for our purposes).
// NOT zombie-aware on its own; see `posixProcessState`.
export function isPidAlive(pid: number, kill: typeof process.kill = process.kill.bind(process)) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false
  }

  try {
    kill(pid, 0)

    return true
  } catch (err: any) {
    return Boolean(err && err.code === 'EPERM')
  }
}

/**
 * Single-letter process state (`ps` style) for a kill(0)-alive pid, or null
 * when it cannot be determined. A ZOMBIE answers signal 0 like a live process
 * (#77259, #120635, #125932). Failures return null (fail-open to alive).
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

function isZombieState(state: string | null | undefined): boolean {
  return Boolean(state && state.toUpperCase().startsWith('Z'))
}

let linuxClockTicks: number | null = null

function linuxCreateTime(pid: number): number | null {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8')
    const fields = stat.slice(stat.lastIndexOf(')') + 1).trim().split(/\s+/)
    // Field 22 of /proc/<pid>/stat (starttime, clock ticks since boot) is index
    // 19 once pid and comm are stripped.
    const ticks = Number(fields[19])
    const btime = Number(/^btime\s+(\d+)/m.exec(fs.readFileSync('/proc/stat', 'utf8'))?.[1])

    if (!Number.isFinite(ticks) || !Number.isFinite(btime)) {
      return null
    }

    if (linuxClockTicks === null) {
      try {
        linuxClockTicks = Number(execFileSync('getconf', ['CLK_TCK'], { encoding: 'utf8', timeout: 2000 }).trim()) || 100
      } catch {
        linuxClockTicks = 100
      }
    }

    return btime + ticks / linuxClockTicks
  } catch {
    return null
  }
}

function darwinCreateTime(pid: number): number | null {
  try {
    const out = execFileSync('ps', ['-o', 'lstart=', '-p', String(pid)], {
      encoding: 'utf8',
      env: { ...process.env, LC_ALL: 'C', LANG: 'C' },
      timeout: 5000
    }).trim()

    const ms = Date.parse(out.replace(/\s+/g, ' '))

    return Number.isFinite(ms) ? ms / 1000 : null
  } catch {
    return null
  }
}

function ownCreateTime(): number | null {
  const ms = (process as { getCreationTime?: () => number | null }).getCreationTime?.()

  return typeof ms === 'number' && Number.isFinite(ms) && ms > 0 ? ms / 1000 : null
}

/**
 * Creation time (unix seconds) of a live pid, or null when it cannot be read.
 * Same sources as the Python/Rust/script writers: Linux /proc starttime +
 * btime, macOS `ps -o lstart=`, Windows `GetProcessTimes` (Electron's own
 * `process.getCreationTime()` for this process; `Get-Process` otherwise).
 */
export async function processCreateTime(pid: number): Promise<number | null> {
  if (!isPidAlive(pid)) {
    return null
  }

  if (process.platform === 'linux') {
    return linuxCreateTime(pid)
  }

  if (process.platform === 'darwin') {
    return darwinCreateTime(pid)
  }

  if (process.platform !== 'win32') {
    return null
  }

  if (pid === process.pid) {
    return ownCreateTime()
  }

  return new Promise(resolve => {
    execFile(
      'powershell.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `(Get-Process -Id ${pid} -ErrorAction Stop).StartTime.ToUniversalTime().Ticks`
      ],
      { encoding: 'utf8', timeout: 15_000, windowsHide: true },
      (error, stdout) => {
        const ticks = Number(String(stdout || '').trim())

        // .NET ticks (100 ns since 0001-01-01) → unix seconds.
        resolve(!error && Number.isFinite(ticks) && ticks > 0 ? (ticks - 621_355_968_000_000_000) / 1e7 : null)
      }
    ).stdin?.end()
  })
}

/**
 * Synchronous variant for module-init callers (desktop-installation's repair
 * lock). On Windows a foreign pid costs one blocking `Get-Process`, so only use
 * it on a rare path.
 */
export function processCreateTimeSync(pid: number): number | null {
  if (!isPidAlive(pid)) {
    return null
  }

  if (process.platform === 'linux') {
    return linuxCreateTime(pid)
  }

  if (process.platform === 'darwin') {
    return darwinCreateTime(pid)
  }

  if (process.platform !== 'win32') {
    return null
  }

  if (pid === process.pid) {
    return ownCreateTime()
  }

  try {
    const ticks = Number(
      execFileSync(
        'powershell.exe',
        [
          '-NoProfile',
          '-NonInteractive',
          '-Command',
          `(Get-Process -Id ${pid} -ErrorAction Stop).StartTime.ToUniversalTime().Ticks`
        ],
        { encoding: 'utf8', timeout: 15_000, windowsHide: true, stdio: ['ignore', 'pipe', 'ignore'] }
      ).trim()
    )

    return Number.isFinite(ticks) && ticks > 0 ? (ticks - 621_355_968_000_000_000) / 1e7 : null
  } catch {
    return null
  }
}

/** C1 rule 3 for a (pid, recorded ct) pair, synchronously. */
export function processIsLiveSync(pid: number, recordedCt: number | null): boolean {
  if (!isPidAlive(pid) || isZombieState(posixProcessState(pid))) {
    return false
  }

  if (recordedCt === null) {
    return true
  }

  const actual = processCreateTimeSync(pid)

  return actual === null || Math.abs(actual - recordedCt) <= CREATE_TIME_TOLERANCE_S
}

export function formatCreateTime(seconds: number): string {
  return seconds.toFixed(3)
}

export interface UpdateMarker {
  pid: number
  /** Unix seconds; null when line 2 is missing/garbled. */
  startedAt: number | null
  /** Recorded owner creation time; null for a v1 marker. */
  ct: number | null
  delegate: { pid: number; ct: number | null } | null
}

const CT_RE = /^ct:(\d+(?:\.\d+)?)$/
const DELEGATE_RE = /^delegate:(\d+)(?:\s+ct:(\d+(?:\.\d+)?))?$/

export function parseUpdateMarker(raw: string): UpdateMarker | null {
  const lines = String(raw).split('\n').map(line => line.trim())

  if (!/^\d+$/.test(lines[0] || '')) {
    return null
  }

  const pid = Number(lines[0])
  const startedAt = /^\d+$/.test(lines[1] || '') ? Number(lines[1]) : null
  let ct: number | null = null
  let delegate: UpdateMarker['delegate'] = null

  for (const line of lines.slice(2)) {
    const ctMatch = CT_RE.exec(line)
    const delegateMatch = DELEGATE_RE.exec(line)

    if (ctMatch && ct === null) {
      ct = Number(ctMatch[1])
    } else if (delegateMatch && !delegate) {
      delegate = { pid: Number(delegateMatch[1]), ct: delegateMatch[2] ? Number(delegateMatch[2]) : null }
    }
  }

  return { pid, startedAt, ct, delegate }
}

export interface MarkerProbeDeps {
  kill?: typeof process.kill
  now?: () => number
  maxAgeMs?: number
  processState?: (pid: number) => string | null
  createTime?: (pid: number) => number | null | Promise<number | null>
}

/** C1 rule 3 for one (pid, recorded ct) pair. Age is the caller's business. */
async function processIsLive(pid: number, recordedCt: number | null, deps: MarkerProbeDeps): Promise<boolean> {
  if (!isPidAlive(pid, deps.kill) || isZombieState((deps.processState || posixProcessState)(pid))) {
    return false
  }

  if (recordedCt === null) {
    return true
  }

  const actual = await (deps.createTime || processCreateTime)(pid)

  return actual === null || Math.abs(actual - recordedCt) <= CREATE_TIME_TOLERANCE_S
}

export type MarkerInspection =
  | { state: 'absent' }
  | { state: 'live'; raw: Buffer; marker: UpdateMarker; ageMs: number; livePid: number }
  | { state: 'dead'; raw: Buffer; marker: UpdateMarker | null }

/** Read and judge the marker without touching it. */
export async function inspectUpdateMarker(hermesHome: string, deps: MarkerProbeDeps = {}): Promise<MarkerInspection> {
  let raw: Buffer

  try {
    raw = fs.readFileSync(markerPath(hermesHome))
  } catch {
    return { state: 'absent' }
  }

  const marker = parseUpdateMarker(raw.toString('utf8'))

  if (!marker) {
    return { state: 'dead', raw, marker: null }
  }

  const now = deps.now || Date.now
  const ageMs = marker.startedAt === null ? Infinity : now() - marker.startedAt * 1000
  const v1Expired = marker.ct === null && ageMs > (deps.maxAgeMs ?? UPDATE_MARKER_MAX_AGE_MS)

  if (!v1Expired && (await processIsLive(marker.pid, marker.ct, deps))) {
    return { state: 'live', raw, marker, ageMs, livePid: marker.pid }
  }

  if (marker.delegate && (await processIsLive(marker.delegate.pid, marker.delegate.ct, deps))) {
    return { state: 'live', raw, marker, ageMs, livePid: marker.delegate.pid }
  }

  return { state: 'dead', raw, marker }
}

/** C1 rule 5: unlink only while the file still holds exactly `expected`. */
export function compareAndDeleteMarker(hermesHome: string, expected: Buffer | string): boolean {
  const file = markerPath(hermesHome)

  try {
    if (!fs.readFileSync(file).equals(Buffer.isBuffer(expected) ? expected : Buffer.from(expected, 'utf8'))) {
      return false
    }

    fs.unlinkSync(file)

    return true
  } catch {
    return false
  }
}

/**
 * `{ pid, ageMs }` while an update is genuinely running, else null. A dead
 * marker is compare-and-deleted so it cannot strand future launches.
 */
export async function readLiveUpdateMarker(hermesHome: string, deps: MarkerProbeDeps = {}) {
  const inspection = await inspectUpdateMarker(hermesHome, deps)

  if (inspection.state === 'live') {
    return {
      pid: inspection.livePid,
      ownerPid: inspection.marker.pid,
      ageMs: inspection.ageMs,
      startedAt: inspection.marker.startedAt
    }
  }

  if (inspection.state === 'dead') {
    compareAndDeleteMarker(hermesHome, inspection.raw)
  }

  return null
}

/**
 * Legacy pre-write for the staged Tauri updater (`hermes-setup.exe --update`):
 * the child IS the updater there, so naming its pid is correct, and frozen
 * builds read only lines 1-2. Exclusive create — a live claim is never
 * overwritten; a dead one is compare-and-deleted first.
 */
export async function writeUpdateMarker(
  hermesHome: string,
  pid: number,
  { startedAt, ...deps }: MarkerProbeDeps & { startedAt?: number } = {}
) {
  const acquiredAt =
    typeof startedAt === 'number' && Number.isInteger(startedAt)
      ? startedAt
      : Math.floor((deps.now || Date.now)() / 1000)

  await createMarkerExclusive(hermesHome, `${pid}\n${acquiredAt}\n`, deps)
}

interface ClaimResult {
  ok: boolean
  body?: string
  /** Set when a LIVE claim blocks ours. */
  owner?: { pid: number; ageMs: number } | null
  error?: string
}

async function createMarkerExclusive(hermesHome: string, body: string, deps: MarkerProbeDeps): Promise<ClaimResult> {
  const file = markerPath(hermesHome)

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const fd = fs.openSync(file, 'wx', 0o644)

      try {
        fs.writeSync(fd, body)
        fs.fsyncSync(fd)
      } finally {
        fs.closeSync(fd)
      }

      return { ok: true, body }
    } catch (error: any) {
      if (error?.code !== 'EEXIST') {
        return { ok: false, owner: null, error: error?.message || String(error) }
      }
    }

    const inspection = await inspectUpdateMarker(hermesHome, deps)

    if (inspection.state === 'live') {
      return { ok: false, owner: { pid: inspection.livePid, ageMs: inspection.ageMs } }
    }

    if (inspection.state === 'dead') {
      compareAndDeleteMarker(hermesHome, inspection.raw)
    }
  }

  const inspection = await inspectUpdateMarker(hermesHome, deps)

  return inspection.state === 'live'
    ? { ok: false, owner: { pid: inspection.livePid, ageMs: inspection.ageMs } }
    : { ok: false, owner: null, error: 'lost the marker claim race twice' }
}

function conflictMessage(owner: { pid: number; ageMs: number }): string {
  const ageMs = Number.isFinite(owner.ageMs) ? Math.max(0, owner.ageMs) : 0
  const mins = Math.floor(ageMs / 60_000)
  const secs = Math.floor((ageMs % 60_000) / 1000)
  const elapsed = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`

  return `An update is already running (PID ${owner.pid}, started ${elapsed} ago). Wait for it to finish, then try again.`
}

/**
 * C2 bridge claim: the Desktop claims the marker in ITS OWN name (pid +
 * creation time) before spawning the hand-off script. Electron is alive until
 * it quits, so the claim never names a short-lived `cmd.exe`/launcher wrapper.
 * A live foreign owner refuses the hand-off (#75778).
 */
export async function claimBridgeMarker(
  hermesHome: string,
  {
    pid = process.pid,
    createTime,
    startedAt,
    ...deps
  }: MarkerProbeDeps & { pid?: number; startedAt?: number } = {}
): Promise<{
  ok: boolean
  body?: string
  conflict?: { pid: number; ageMs: number; message: string } | null
  error?: string
}> {
  const ct = await (createTime || processCreateTime)(pid)
  const acquiredAt = Number.isInteger(startedAt) ? startedAt : Math.floor((deps.now || Date.now)() / 1000)
  const body = `${pid}\n${acquiredAt}\n${ct === null ? '' : `ct:${formatCreateTime(ct)}\n`}`

  fs.mkdirSync(hermesHome, { recursive: true })
  const result = await createMarkerExclusive(hermesHome, body, { ...deps, createTime })

  if (result.ok || !result.owner) {
    return result
  }

  return { ok: false, conflict: { ...result.owner, message: conflictMessage(result.owner) } }
}

/**
 * C2: the hand-off has started only once the script took the marker — its
 * owner pid is no longer ours. A wrapper's exit code says nothing about
 * whether the script ever ran (#66753: CLM-blocked Add-Type, a missing
 * /usr/bin/python3 for the posix daemonizer).
 */
export async function waitForHandoffClaim(
  hermesHome: string,
  ownPid: number,
  {
    timeoutMs = HANDOFF_CLAIM_TIMEOUT_MS,
    pollMs = 200,
    now = Date.now,
    sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))
  }: { timeoutMs?: number; pollMs?: number; now?: () => number; sleep?: (ms: number) => Promise<void> } = {}
): Promise<{ taken: true; pid: number } | { taken: false }> {
  const deadline = now() + timeoutMs

  for (;;) {
    let marker: UpdateMarker | null = null

    try {
      marker = parseUpdateMarker(fs.readFileSync(markerPath(hermesHome), 'utf8'))
    } catch {
      marker = null
    }

    if (marker && marker.pid !== ownPid) {
      return { taken: true, pid: marker.pid }
    }

    if (now() >= deadline) {
      return { taken: false }
    }

    await sleep(pollMs)
  }
}

/**
 * Whether a NEW updater hand-off must be refused because a different, live
 * updater owns the marker (#75778). Null when it is safe to spawn.
 */
export async function updateHandoffConflict(hermesHome: string, deps: MarkerProbeDeps = {}) {
  const owner = await readLiveUpdateMarker(hermesHome, deps)

  return owner ? { pid: owner.pid, ageMs: owner.ageMs, message: conflictMessage(owner) } : null
}
