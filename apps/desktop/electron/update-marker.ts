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
 * Parsing is positional and identical in every language (contract A2): a
 * leading BOM and `\r\n` are accepted; line 2 must be an integer or the marker
 * is malformed (dead); a line 3 that is not `ct:<n>` makes it v1; a line 4
 * that is not `delegate:<pid> ct:<n>` is ignored.
 *
 * Liveness: a pid that is alive (not a zombie) with a creation time MATCHING
 * the recorded one within 2 s is live regardless of age — a Windows update is
 * routinely 25-40 minutes. A live pid whose creation time is unknown (v1
 * marker, or the probe was refused) is live only inside the legacy 20-minute
 * ceiling (A1): without a creation time nothing else tells a reused pid from
 * the original owner. The marker is LIVE when the owner OR the delegate
 * (`hermes update` running under a hand-off script's claim) is live, so a
 * killed script cannot hide a still-mutating update.
 *
 * Claims publish a complete body with an exclusive hard link (A3), so a
 * reader never sees a half-written claim; where links are unsupported the
 * O_EXCL fallback can expose an empty file for an instant, and readers treat
 * a 0-byte marker younger than 5 s as live.
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

/** A 0-byte marker this young is a claim being written, not garbage (A3). */
export const EMPTY_MARKER_GRACE_MS = 5_000

/** .NET ticks (100 ns since 0001-01-01) at the unix epoch. */
const DOTNET_UNIX_EPOCH_TICKS = 621_355_968_000_000_000

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

/**
 * `ps -o lstart=` creation time (the macOS source; procps prints the same).
 * Printed in UTC and parsed as UTC (A1): local time is ambiguous for an hour
 * at every DST fall-back and can put a live owner 3600 s off its record.
 */
export function psCreateTime(pid: number): number | null {
  try {
    const out = execFileSync('ps', ['-o', 'lstart=', '-p', String(pid)], {
      encoding: 'utf8',
      env: { ...process.env, LC_ALL: 'C', LANG: 'C', TZ: 'UTC0' },
      timeout: 5000
    }).trim()

    const ms = out ? Date.parse(`${out.replace(/\s+/g, ' ')} GMT`) : NaN

    return Number.isFinite(ms) ? ms / 1000 : null
  } catch {
    return null
  }
}

/**
 * Windows creation time through CIM (A1): `Win32_Process.CreationDate` is read
 * with limited query rights, so it answers for SYSTEM, elevated and other-user
 * processes where `Get-Process .StartTime` is access-denied.
 */
function windowsCreateTimeCommand(pid: number): string[] {
  return [
    '-NoProfile',
    '-NonInteractive',
    '-Command',
    `(Get-CimInstance Win32_Process -Filter 'ProcessId=${pid}' -ErrorAction Stop).CreationDate.ToUniversalTime().Ticks`
  ]
}

function dotnetTicksToUnix(stdout: unknown): number | null {
  const ticks = Number(String(stdout ?? '').trim())

  return Number.isFinite(ticks) && ticks > 0 ? (ticks - DOTNET_UNIX_EPOCH_TICKS) / 1e7 : null
}

function ownCreateTime(): number | null {
  const ms = (process as { getCreationTime?: () => number | null }).getCreationTime?.()

  return typeof ms === 'number' && Number.isFinite(ms) && ms > 0 ? ms / 1000 : null
}

/**
 * Creation time (unix seconds) of a live pid, or null when it cannot be read.
 * Same sources as the Python/Rust/script writers: Linux /proc starttime +
 * btime, macOS `ps -o lstart=` (UTC), Windows `GetProcessTimes` (Electron's own
 * `process.getCreationTime()` for this process; CIM `CreationDate` otherwise).
 * A Windows foreign pid costs one powershell spawn: pollers wrap this in
 * `cachedCreateTimeProbe`.
 */
export async function processCreateTime(pid: number): Promise<number | null> {
  if (!isPidAlive(pid)) {
    return null
  }

  if (process.platform !== 'win32' || (pid === process.pid && ownCreateTime() !== null)) {
    return processCreateTimeSync(pid)
  }

  return new Promise(resolve => {
    execFile(
      'powershell.exe',
      windowsCreateTimeCommand(pid),
      { encoding: 'utf8', timeout: 15_000, windowsHide: true },
      (error, stdout) => resolve(error ? null : dotnetTicksToUnix(stdout))
    ).stdin?.end()
  })
}

/**
 * Synchronous variant for module-init callers (desktop-installation's repair
 * lock). On Windows a foreign pid costs one blocking powershell spawn, so only
 * use it on a rare path.
 */
export function processCreateTimeSync(pid: number): number | null {
  if (!isPidAlive(pid)) {
    return null
  }

  if (process.platform === 'linux') {
    return linuxCreateTime(pid)
  }

  if (process.platform === 'darwin') {
    return psCreateTime(pid)
  }

  if (process.platform !== 'win32') {
    return null
  }

  // Electron's own GetProcessTimes; plain Node (tests, tooling) has no
  // getCreationTime and falls through to CIM like any other pid.
  const own = pid === process.pid ? ownCreateTime() : null

  if (own !== null) {
    return own
  }

  try {
    return dotnetTicksToUnix(
      execFileSync('powershell.exe', windowsCreateTimeCommand(pid), {
        encoding: 'utf8',
        timeout: 15_000,
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'ignore']
      })
    )
  } catch {
    return null
  }
}

export type CreateTimeProbe = (pid: number) => number | null | Promise<number | null>

/**
 * One creation-time probe per pid for the life of ONE waiter (A1) — on Windows
 * each probe is a powershell spawn, and the update gate polls every second.
 * Scope it to a single wait, never module-wide: only a fresh probe tells a
 * pid reused since the last wait from the owner it replaced.
 */
export function cachedCreateTimeProbe(probe: CreateTimeProbe = processCreateTime): CreateTimeProbe {
  const memo = new Map<number, Promise<number | null>>()

  return pid => {
    let hit = memo.get(pid)

    if (!hit) {
      hit = Promise.resolve(probe(pid))
      memo.set(pid, hit)
    }

    return hit
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
  /** Unix seconds (line 2; an integer, or the marker is malformed). */
  startedAt: number
  /** Recorded owner creation time; null for a v1 marker. */
  ct: number | null
  delegate: { pid: number; ct: number } | null
}

const INT_LINE_RE = /^\d+$/
const CT_LINE_RE = /^ct:(\d+(?:\.\d+)?)$/
const DELEGATE_LINE_RE = /^delegate:(\d+) ct:(\d+(?:\.\d+)?)$/

/**
 * Positional parse, identical in every reader (A2). Null = MALFORMED (dead):
 * line 1 not a pid or line 2 not an integer. A bad line 3 means v1; a bad
 * line 4 is ignored.
 */
export function parseUpdateMarker(raw: string): UpdateMarker | null {
  const lines = String(raw)
    .replace(/^\uFEFF/, '')
    .split('\n')
    .map(line => line.trim())

  const [pidLine = '', startedLine = '', ctLine = '', delegateLine = ''] = lines

  if (!INT_LINE_RE.test(pidLine) || !INT_LINE_RE.test(startedLine)) {
    return null
  }

  const ct = CT_LINE_RE.exec(ctLine)
  const delegate = DELEGATE_LINE_RE.exec(delegateLine)

  return {
    pid: Number(pidLine),
    startedAt: Number(startedLine),
    ct: ct ? Number(ct[1]) : null,
    delegate: delegate ? { pid: Number(delegate[1]), ct: Number(delegate[2]) } : null
  }
}

export interface MarkerProbeDeps {
  kill?: typeof process.kill
  now?: () => number
  maxAgeMs?: number
  processState?: (pid: number) => string | null
  createTime?: CreateTimeProbe
}

/**
 * C1 rule 3 identity of one (pid, recorded ct) pair: `match` = alive with the
 * recorded creation time; `unknown` = alive but no creation time to compare
 * (v1, or the probe was refused); `dead` = gone, a zombie, or a reused pid.
 */
async function processIdentity(
  pid: number,
  recordedCt: number | null,
  deps: MarkerProbeDeps
): Promise<'match' | 'unknown' | 'dead'> {
  if (!isPidAlive(pid, deps.kill) || isZombieState((deps.processState || posixProcessState)(pid))) {
    return 'dead'
  }

  if (recordedCt === null) {
    return 'unknown'
  }

  const actual = await (deps.createTime || processCreateTime)(pid)

  if (actual === null) {
    return 'unknown'
  }

  return Math.abs(actual - recordedCt) <= CREATE_TIME_TOLERANCE_S ? 'match' : 'dead'
}

export type MarkerInspection =
  | { state: 'absent' }
  /** `marker` is null for a 0-byte claim still being written (A3). */
  | { state: 'live'; raw: Buffer; marker: UpdateMarker | null; ageMs: number; livePid: number }
  | { state: 'dead'; raw: Buffer; marker: UpdateMarker | null }

/** Read and judge the marker without touching it. */
export async function inspectUpdateMarker(hermesHome: string, deps: MarkerProbeDeps = {}): Promise<MarkerInspection> {
  const file = markerPath(hermesHome)
  const now = deps.now || Date.now
  let raw: Buffer

  try {
    raw = fs.readFileSync(file)
  } catch {
    return { state: 'absent' }
  }

  if (raw.length === 0) {
    let ageMs = Infinity

    try {
      ageMs = now() - fs.statSync(file).mtimeMs
    } catch {
      // Gone since the read: nothing to judge.
    }

    return ageMs < EMPTY_MARKER_GRACE_MS
      ? { state: 'live', raw, marker: null, ageMs: Math.max(0, ageMs), livePid: 0 }
      : { state: 'dead', raw, marker: null }
  }

  const marker = parseUpdateMarker(raw.toString('utf8'))

  if (!marker) {
    return { state: 'dead', raw, marker: null }
  }

  const ageMs = now() - marker.startedAt * 1000
  const withinCeiling = ageMs <= (deps.maxAgeMs ?? UPDATE_MARKER_MAX_AGE_MS)

  // A1: only a MATCHING creation time is live regardless of age.
  const isLive = (identity: 'match' | 'unknown' | 'dead') =>
    identity === 'match' || (identity === 'unknown' && withinCeiling)

  if (isLive(await processIdentity(marker.pid, marker.ct, deps))) {
    return { state: 'live', raw, marker, ageMs, livePid: marker.pid }
  }

  if (marker.delegate && isLive(await processIdentity(marker.delegate.pid, marker.delegate.ct, deps))) {
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
      ownerPid: inspection.marker?.pid ?? 0,
      ageMs: inspection.ageMs,
      startedAt: inspection.marker?.startedAt ?? null
    }
  }

  if (inspection.state === 'dead') {
    compareAndDeleteMarker(hermesHome, inspection.raw)
  }

  return null
}

function markerBody(pid: number, startedAt: number, ct: number | null): string {
  return `${pid}\n${startedAt}\n${ct === null ? '' : `ct:${formatCreateTime(ct)}\n`}`
}

/**
 * Pre-write for the staged Tauri updater (`hermes-setup.exe --update`): the
 * child IS the updater there, so the marker names its pid AND its creation
 * time. A v2 body takes this path off the v1 20-minute ceiling, lets the
 * updater adopt it as its own claim, and lets the `hermes update` it runs add
 * its delegate line (frozen updaters read only lines 1-2). An unreadable ct
 * falls back to a v1 body. Exclusive publish — a live claim is never
 * overwritten; a dead one is compare-and-deleted first.
 */
export async function writeUpdateMarker(
  hermesHome: string,
  pid: number,
  { startedAt, ...deps }: MarkerProbeDeps & { startedAt?: number } = {}
): Promise<ClaimResult> {
  const acquiredAt =
    typeof startedAt === 'number' && Number.isInteger(startedAt)
      ? startedAt
      : Math.floor((deps.now || Date.now)() / 1000)

  const ct = await (deps.createTime || processCreateTime)(pid)

  return createMarkerExclusive(hermesHome, markerBody(pid, acquiredAt, ct), deps)
}

interface ClaimResult {
  ok: boolean
  body?: string
  /** Set when a LIVE claim blocks ours. */
  owner?: { pid: number; ageMs: number } | null
  error?: string
}

const TMP_SIBLING_RE = /^\.hermes-update-in-progress\.(\d+)(?:\.\d+)?\.tmp$/
let tmpSequence = 0

/** Drop publish/CAS tmp siblings whose writer died between write and link/rename. */
function reclaimDeadTmpSiblings(hermesHome: string) {
  try {
    for (const name of fs.readdirSync(hermesHome)) {
      const pid = Number(TMP_SIBLING_RE.exec(name)?.[1])

      if (pid && !isPidAlive(pid)) {
        fs.rmSync(path.join(hermesHome, name), { force: true })
      }
    }
  } catch {
    // Best effort: litter never blocks a claim.
  }
}

/**
 * Publish `body` at `file` only if nothing is there (A3): write a complete tmp
 * sibling, then hard-link it into place — link(2)/CreateHardLink fail with
 * EEXIST instead of replacing, and a reader never sees a partial body. A
 * filesystem without hard links falls back to O_EXCL create + write.
 */
function publishExclusive(file: string, body: string): 'published' | 'exists' {
  const tmp = `${file}.${process.pid}.${++tmpSequence}.tmp`
  let linked: boolean

  try {
    const fd = fs.openSync(tmp, 'wx', 0o644)

    try {
      fs.writeSync(fd, body)
      fs.fsyncSync(fd)
    } finally {
      fs.closeSync(fd)
    }

    try {
      fs.linkSync(tmp, file)
      linked = true
    } catch (error: any) {
      if (error?.code === 'EEXIST') {
        return 'exists'
      }

      linked = false
    }
  } finally {
    fs.rmSync(tmp, { force: true })
  }

  if (linked) {
    return 'published'
  }

  let fd: number

  try {
    fd = fs.openSync(file, 'wx', 0o644)
  } catch (error: any) {
    if (error?.code === 'EEXIST') {
      return 'exists'
    }

    throw error
  }

  try {
    fs.writeSync(fd, body)
    fs.fsyncSync(fd)
  } finally {
    fs.closeSync(fd)
  }

  return 'published'
}

async function createMarkerExclusive(hermesHome: string, body: string, deps: MarkerProbeDeps): Promise<ClaimResult> {
  const file = markerPath(hermesHome)

  reclaimDeadTmpSiblings(hermesHome)

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      if (publishExclusive(file, body) === 'published') {
        return { ok: true, body }
      }
    } catch (error: any) {
      return { ok: false, owner: null, error: error?.message || String(error) }
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
  if (!owner.pid) {
    return 'Another update is starting right now. Wait for it to finish, then try again.'
  }

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

  const acquiredAt =
    typeof startedAt === 'number' && Number.isInteger(startedAt)
      ? startedAt
      : Math.floor((deps.now || Date.now)() / 1000)

  const body = markerBody(pid, acquiredAt, ct)

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
