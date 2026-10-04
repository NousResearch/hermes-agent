/**
 * The update marker `HERMES_HOME/.hermes-update-in-progress` (#50238), format v2.
 *
 * One file, shared by every updater (Python `hermes_cli/update_lock.py`, the Rust
 * `UpdateMarkerGuard`, `scripts/desktop-update/{windows.ps1,posix.sh}`) and this
 * Desktop. While it names a LIVE owner, a Desktop reopened mid-update must not
 * spawn a backend on the runtime being replaced, and no second update may start.
 *
 * Parsing and judging live in `update-marker-judge.ts` and follow the shared
 * corpus `tests/fixtures/update_marker_corpus.json` (A7 rules 4 and 7).
 *
 * A7 rule 3: this Desktop NEVER deletes or rewrites the marker. Node has no
 * portable kernel lock, so every read-judge-mutate (reclaim a dead claim,
 * withdraw a bridge) belongs to the checkout's script helper, which holds the
 * `<marker>.lock` sidecar for the whole decision (see
 * `updater/marker-helper.ts`). Electron reads, decides "running / not
 * running", and may exclusive-create its own bridge or the staged updater's
 * pre-write — an exclusive create can only succeed where no marker exists.
 *
 * Claims publish a complete body with an exclusive hard link (A3), so a
 * reader never sees a half-written claim; where links are unsupported the
 * O_EXCL fallback can expose an empty file for an instant, and readers treat
 * a 0-byte marker younger than 5 s as live.
 */

import fs from 'fs'
import { execFile, execFileSync } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import path from 'path'

import {
  CREATE_TIME_TOLERANCE_S,
  hasForeignLiveIdentity,
  type JudgeEnv,
  judgeMarkerText,
  type MarkerJudgement,
  RUN_ID_RE,
  type UpdateMarker,
  V1_MAX_AGE_S
} from './update-marker-judge'

export {
  CREATE_TIME_TOLERANCE_S,
  type IdentityState,
  judgeMarkerText,
  type MarkerJudgement,
  OWN_CT_EPSILON_S,
  parseUpdateMarker,
  type UpdateMarker,
  V1_MAX_AGE_S
} from './update-marker-judge'

/** Legacy ceiling, applied ONLY to identities without a comparable creation time. */
export const UPDATE_MARKER_MAX_AGE_MS = V1_MAX_AGE_S * 1000

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

export interface MarkerProbeDeps {
  kill?: typeof process.kill
  /** Milliseconds since the epoch. */
  now?: () => number
  processState?: (pid: number) => string | null
  createTime?: CreateTimeProbe
  /** The pid judged as "us" (A7 rule 4); default this process. */
  ownPid?: number
}

/** The host's process facts as a judge environment (A7 rule 4). */
export function hostJudgeEnv(deps: MarkerProbeDeps = {}): JudgeEnv {
  const createTime = deps.createTime || processCreateTime
  const ourPid = deps.ownPid ?? process.pid

  return {
    ourPid,
    ourCt: () => createTime(ourPid),
    isAlive: pid => isPidAlive(pid, deps.kill) && !isZombieState((deps.processState || posixProcessState)(pid)),
    createTime,
    nowS: (deps.now || Date.now)() / 1000
  }
}

export type MarkerInspection =
  | { state: 'absent' }
  /**
   * Someone else's update is running (a live identity that is not this
   * process), or a 0-byte claim is being written (`judgement` null, A3).
   */
  | { state: 'live'; raw: Buffer; marker: UpdateMarker | null; ageMs: number; livePid: number; judgement: MarkerJudgement | null }
  /** A bridge of THIS process incarnation and nothing foreign alive in it. */
  | { state: 'ours'; raw: Buffer; marker: UpdateMarker; judgement: MarkerJudgement }
  /** Dead or malformed (`judgement` null for a stale 0-byte file). */
  | { state: 'dead'; raw: Buffer; marker: UpdateMarker | null; judgement: MarkerJudgement | null }

/** Read and judge the marker. Never touches it. */
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
      ? { state: 'live', raw, marker: null, ageMs: Math.max(0, ageMs), livePid: 0, judgement: null }
      : { state: 'dead', raw, marker: null, judgement: null }
  }

  const judgement = await judgeMarkerText(raw.toString('utf8'), hostJudgeEnv(deps))
  const { marker } = judgement

  if (!marker || judgement.verdict === 'malformed' || judgement.verdict === 'dead') {
    return { state: 'dead', raw, marker, judgement }
  }

  if (hasForeignLiveIdentity(judgement)) {
    const ownerForeign = judgement.ownerState === 'match' || judgement.ownerState === 'unknown'
    const livePid = ownerForeign ? marker.pid : (marker.delegate?.pid ?? marker.pid)

    return { state: 'live', raw, marker, ageMs: now() - marker.startedAt * 1000, livePid, judgement }
  }

  return { state: 'ours', raw, marker, judgement }
}

/**
 * `{ pid, ageMs }` while someone else's update is genuinely running, else
 * null. Read-only (A7 rule 3): a dead marker is left for the script helper /
 * the next claimant to reclaim under the lock.
 */
export async function readLiveUpdateMarker(hermesHome: string, deps: MarkerProbeDeps = {}) {
  const inspection = await inspectUpdateMarker(hermesHome, deps)

  if (inspection.state !== 'live') {
    return null
  }

  return {
    pid: inspection.livePid,
    ownerPid: inspection.marker?.pid ?? 0,
    ageMs: inspection.ageMs,
    startedAt: inspection.marker?.startedAt ?? null
  }
}

function markerBody(pid: number, startedAt: number, ct: number | null, runId?: string | null): string {
  const ctLine = ct === null ? '' : `ct:${formatCreateTime(ct)}\n`
  const runLine = runId ? `run:${runId}\n` : ''

  return `${pid}\n${startedAt}\n${ctLine}${runLine}`
}

/** What an exclusive create found instead of publishing. */
export interface ExistingMarker {
  state: 'dead' | 'ours'
  raw: Buffer
  /** The run id the existing claim carries (to withdraw a stale bridge of ours). */
  run: string | null
}

export interface ClaimResult {
  ok: boolean
  body?: string
  /** Set when a LIVE claim blocks ours. */
  owner?: { pid: number; ageMs: number } | null
  /** Set when a dead/malformed claim, or a stale bridge of this process, blocks ours. */
  existing?: ExistingMarker
  error?: string
}

/**
 * Pre-write for the staged Tauri updater (`hermes-setup.exe --update`): the
 * child IS the updater there, so the marker names its pid AND its creation
 * time. A v2 body takes this path off the v1 20-minute ceiling, lets the
 * updater adopt it as its own claim, and lets the `hermes update` it runs add
 * its delegate line. Exclusive create ONLY (A7 rule 3): over any existing
 * marker — live or dead — the pre-write is skipped and the staged updater
 * claims (and reclaims) for itself.
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

/** Log line for a staged-updater pre-write that was skipped (exclusive create lost). */
export function describeSkippedPrewrite(result: ClaimResult): string {
  if (result.owner) {
    return `an update owned by pid ${result.owner.pid || '(claim being written)'} holds the marker`
  }

  if (result.existing) {
    return `a ${result.existing.state} marker exists; the staged updater reclaims it under its own lock`
  }

  return result.error || 'unknown error'
}

const TMP_SIBLING_RE = /^\.hermes-update-in-progress\.(\d+)(?:\.\d+)?\.tmp$/
let tmpSequence = 0

/**
 * Drop OUR publish tmp siblings (`<marker>.<pid>.<seq>.tmp`) whose writer died
 * between write and link. Never the marker itself.
 */
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

/**
 * One exclusive create (A7 rule 2). On EEXIST the existing marker is judged
 * read-only and reported: a live owner blocks, a dead one / our own stale
 * bridge is handed back for the caller to route through the script helper.
 */
async function createMarkerExclusive(hermesHome: string, body: string, deps: MarkerProbeDeps): Promise<ClaimResult> {
  const file = markerPath(hermesHome)

  reclaimDeadTmpSiblings(hermesHome)

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

  if (inspection.state === 'absent') {
    // Withdrawn between our create and our read: the caller may retry once.
    return { ok: false, owner: null, error: 'the marker changed while it was being claimed' }
  }

  return {
    ok: false,
    owner: null,
    existing: { state: inspection.state, raw: inspection.raw, run: inspection.judgement?.run ?? null }
  }
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

/** A protocol-2 hand-off run id: `desk-<pid>-<base36 ms>-<hex4>` (run-line grammar). */
export function makeHandoffRunId(pid: number = process.pid, nowMs: number = Date.now()): string {
  const id = `desk-${pid}-${nowMs.toString(36)}-${randomBytes(2).toString('hex')}`

  if (!RUN_ID_RE.test(id)) {
    throw new Error(`invalid hand-off run id ${id}`)
  }

  return id
}

export interface BridgeClaimResult {
  ok: boolean
  body?: string
  /** A live foreign owner (or a claim being written) refuses the hand-off. */
  conflict?: { pid: number; ageMs: number; message: string } | null
  /** A dead/malformed marker or a stale bridge of ours blocks the create (protocol 2: script helper). */
  existing?: ExistingMarker
  error?: string
}

/**
 * C2 bridge claim, protocol 2 (SPEC 4a): the Desktop exclusive-creates the
 * marker in ITS OWN name (pid + creation time) with the hand-off `run:` line
 * before spawning the script. Never reclaims or overwrites anything (A7
 * rule 3): an existing marker is reported for the caller to route through
 * the checkout's script helper.
 */
export async function claimBridgeMarker(
  hermesHome: string,
  {
    pid = process.pid,
    createTime,
    startedAt,
    runId,
    ...deps
  }: MarkerProbeDeps & { pid?: number; startedAt?: number; runId?: string } = {}
): Promise<BridgeClaimResult> {
  const ct = await (createTime || processCreateTime)(pid)

  const acquiredAt =
    typeof startedAt === 'number' && Number.isInteger(startedAt)
      ? startedAt
      : Math.floor((deps.now || Date.now)() / 1000)

  const body = markerBody(pid, acquiredAt, ct, runId)

  fs.mkdirSync(hermesHome, { recursive: true })
  const result = await createMarkerExclusive(hermesHome, body, { ...deps, createTime, ownPid: deps.ownPid ?? pid })

  if (result.ok || !result.owner) {
    return { ok: result.ok, body: result.body, existing: result.existing, error: result.error }
  }

  return { ok: false, conflict: { ...result.owner, message: conflictMessage(result.owner) } }
}

export interface HandoffClaimOptions extends MarkerProbeDeps {
  /** Protocol 2: the run id passed to the script; the claim must carry it. */
  runId?: string
  /** Legacy scripts: the HERMES_UPDATE_STARTED_AT they echo on line 2. */
  startedAt?: number
  timeoutMs?: number
  pollMs?: number
  sleep?: (ms: number) => Promise<void>
}

/**
 * Whether the marker body proves a LIVE successor took the hand-off (R5,
 * A7 rule 6). Never the Desktop's own incarnation, never a dead or reused pid.
 * - protocol 2 (`runId`): line-1 owner verified by pid AND creation time
 *   (`match`, not `unknown`) and the claim's run == runId;
 * - legacy (`startedAt`): line-1 owner live (ct verified when recorded, else
 *   alive inside the v1 ceiling) and line 2 == the startedAt we passed.
 */
export async function handoffTakenBy(
  text: string,
  desktopPid: number,
  { runId, startedAt, ...deps }: HandoffClaimOptions
): Promise<number | null> {
  const judgement = await judgeMarkerText(text, hostJudgeEnv({ ...deps, ownPid: desktopPid }))
  const marker = judgement.marker

  if (!marker || marker.pid === desktopPid) {
    return null
  }

  if (runId !== undefined) {
    return judgement.ownerState === 'match' && marker.run === runId ? marker.pid : null
  }

  const live = judgement.ownerState === 'match' || judgement.ownerState === 'unknown'

  return live && (startedAt === undefined || marker.startedAt === startedAt) ? marker.pid : null
}

/**
 * C2 + R5: the hand-off has started only once a LIVE, correlated script
 * process holds the marker. A wrapper's exit code says nothing about whether
 * the script ever ran (#66753), and a pid that claimed then died is not a
 * running update.
 */
export async function waitForHandoffClaim(
  hermesHome: string,
  desktopPid: number,
  {
    timeoutMs = HANDOFF_CLAIM_TIMEOUT_MS,
    pollMs = 200,
    now = Date.now,
    sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms)),
    createTime,
    ...rest
  }: HandoffClaimOptions = {}
): Promise<{ taken: true; pid: number } | { taken: false }> {
  const deadline = now() + timeoutMs
  // One creation-time probe per pid for this wait (a powershell spawn on Windows).
  const probe = cachedCreateTimeProbe(createTime)

  for (;;) {
    let text: string | null = null

    try {
      text = fs.readFileSync(markerPath(hermesHome), 'utf8')
    } catch {
      text = null
    }

    const pid = text ? await handoffTakenBy(text, desktopPid, { ...rest, now, createTime: probe }) : null

    if (pid !== null) {
      return { taken: true, pid }
    }

    if (now() >= deadline) {
      return { taken: false }
    }

    await sleep(pollMs)
  }
}

/**
 * Whether a NEW updater hand-off must be refused because a different, live
 * updater owns the marker (#75778). Null when it is safe to spawn. Read-only.
 */
export async function updateHandoffConflict(hermesHome: string, deps: MarkerProbeDeps = {}) {
  const owner = await readLiveUpdateMarker(hermesHome, deps)

  return owner ? { pid: owner.pid, ageMs: owner.ageMs, message: conflictMessage(owner) } : null
}
