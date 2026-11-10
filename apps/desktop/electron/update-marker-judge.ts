/**
 * Pure parse + judge of the update marker `HERMES_HOME/.hermes-update-in-progress`
 * (A7 rule 4, contract in `tests/fixtures/update_marker_corpus.json`).
 *
 * The corpus is authoritative: every reader (Python, Rust, bash, PowerShell,
 * this Desktop) runs its `judge` cases. Process facts are injected
 * (`JudgeEnv`), so the same code serves the corpus table and the real host.
 *
 * Format:
 *
 *     line 1   owner pid, ASCII digits, fits u32 (else MALFORMED). pid 0 is never a process.
 *     line 2   started_at unix seconds, ASCII digits, fits u64 (else MALFORMED; any digit count)
 *     line 3   ct:<digits[.digits]> owner creation time; anything else => v1
 *     lines 4+ 'delegate:<pid> ct:<ct>' (exactly one space; first well-formed wins)
 *              'run:<[A-Za-z0-9._-]{1,128}>' (first wins); other lines ignored
 *
 * Per line: one leading BOM (line 1), one trailing CR, then surrounding
 * spaces/tabs are dropped. Interior whitespace in a pid makes the line bad.
 */

/** A claim naming OUR pid is ours only within this creation-time slack (rounding to 3 dp). */
export const OWN_CT_EPSILON_S = 0.005

/** |recorded - actual| creation-time slack for any other pid, in seconds. */
export const CREATE_TIME_TOLERANCE_S = 2.0

/** Ceiling for an identity whose creation time is unknown (v1 marker, or unreadable). */
export const V1_MAX_AGE_S = 1200

const U32_MAX = 0xffff_ffff
const U64_MAX = 18_446_744_073_709_551_615n
const INT_RE = /^[0-9]+$/
const CT_RE = /^ct:([0-9]+(?:\.[0-9]+)?)$/
const DELEGATE_RE = /^delegate:([0-9]+) ct:([0-9]+(?:\.[0-9]+)?)$/
const RUN_RE = /^run:([A-Za-z0-9._-]{1,128})$/

/** The run-id grammar of a `run:` line. */
export const RUN_ID_RE = /^[A-Za-z0-9._-]{1,128}$/

export interface UpdateMarker {
  pid: number
  /** Unix seconds (line 2). */
  startedAt: number
  /** Recorded owner creation time; null for a v1 marker. */
  ct: number | null
  delegate: { pid: number; ct: number } | null
  /** The hand-off run id this claim carries (protocol 2), or null. */
  run: string | null
}

function cleanLine(line: string): string {
  return line.replace(/\r$/, '').replace(/^[ \t]+|[ \t]+$/g, '')
}

function u32(text: string): number | null {
  const value = Number(text)

  return Number.isSafeInteger(value) && value <= U32_MAX ? value : null
}

/** Positional parse, identical in every reader. Null = MALFORMED. */
export function parseUpdateMarker(raw: string): UpdateMarker | null {
  const lines = String(raw)
    .replace(/^\uFEFF/, '')
    .split('\n')
    .map(cleanLine)

  if (lines.length < 2 || !INT_RE.test(lines[0]) || !INT_RE.test(lines[1])) {
    return null
  }

  const pid = u32(lines[0])

  // BigInt parses any digit string exactly (INT_RE above) and never throws here.
  if (pid === null || BigInt(lines[1]) > U64_MAX) {
    return null
  }

  const ct = lines.length > 2 ? CT_RE.exec(lines[2]) : null
  let delegate: UpdateMarker['delegate'] = null
  let run: string | null = null

  for (const line of lines.slice(3)) {
    const d = delegate === null ? DELEGATE_RE.exec(line) : null
    const delegatePid = d ? u32(d[1]) : null

    if (d && delegatePid !== null) {
      delegate = { pid: delegatePid, ct: Number(d[2]) }

      continue
    }

    const r = run === null ? RUN_RE.exec(line) : null

    if (r) {
      run = r[1]
    }
  }

  return { pid, startedAt: Number(lines[1]), ct: ct ? Number(ct[1]) : null, delegate, run }
}

/**
 * State of one (pid, recorded ct) identity:
 * - `ours`: our own pid at our exact incarnation (5 ms);
 * - `match`: another live pid whose creation time matches within 2 s;
 * - `unknown`: another live pid with no ct to compare (v1 or unreadable), inside the v1 ceiling;
 * - `dead`: gone, zombie, pid 0, reused pid, past the ceiling, or a previous incarnation of us.
 */
export type IdentityState = 'ours' | 'match' | 'unknown' | 'dead'

export type MarkerVerdict = 'malformed' | 'dead' | 'ours' | 'live'

export interface MarkerJudgement {
  verdict: MarkerVerdict
  /** Owner identity if live, else the delegate if live, else null. */
  owner: number | null
  run: string | null
  marker: UpdateMarker | null
  ownerState: IdentityState
  /** `null` when the marker has no delegate line. */
  delegateState: IdentityState | null
}

export interface JudgeEnv {
  ourPid: number
  /** Our own creation time (unix seconds); null when unreadable. */
  ourCt: () => number | null | Promise<number | null>
  /** Alive and not a zombie. Never called for our own pid. */
  isAlive: (pid: number) => boolean | Promise<boolean>
  /** Creation time of a live pid; null when unreadable. Only called when the marker recorded one. */
  createTime: (pid: number) => number | null | Promise<number | null>
  /** Unix seconds. */
  nowS: number
}

/** {@link JudgeEnv} whose process facts answer synchronously. */
export interface SyncJudgeEnv {
  ourPid: number
  ourCt: () => number | null
  isAlive: (pid: number) => boolean
  createTime: (pid: number) => number | null
  nowS: number
}

/**
 * The identity rule itself, the one copy every Desktop reader runs: the
 * async judge resolves the facts first, the module-init repair lock reads
 * them synchronously. `startedAt` anchors the v1 ceiling.
 */
export function identityStateSync(
  pid: number,
  recordedCt: number | null,
  startedAt: number,
  env: SyncJudgeEnv
): IdentityState {
  if (pid === 0) {
    return 'dead'
  }

  if (pid === env.ourPid) {
    // A7 rule 4: our pid is ours only at our exact incarnation; a no-ct claim
    // naming our pid is a previous incarnation, never live.
    const ourCt = env.ourCt()

    return recordedCt !== null && ourCt !== null && Math.abs(recordedCt - ourCt) <= OWN_CT_EPSILON_S ? 'ours' : 'dead'
  }

  if (!env.isAlive(pid)) {
    return 'dead'
  }

  const actual = recordedCt === null ? null : env.createTime(pid)

  if (recordedCt === null || actual === null) {
    return env.nowS - startedAt <= V1_MAX_AGE_S ? 'unknown' : 'dead'
  }

  return Math.abs(recordedCt - actual) <= CREATE_TIME_TOLERANCE_S ? 'match' : 'dead'
}

/** Read exactly the facts {@link identityStateSync} consults, then apply it. */
async function identityState(
  pid: number,
  recordedCt: number | null,
  startedAt: number,
  env: JudgeEnv
): Promise<IdentityState> {
  const own = pid !== 0 && pid === env.ourPid
  const ourCt = own ? await env.ourCt() : null
  const alive = pid !== 0 && !own && (await env.isAlive(pid))
  const actual = alive && recordedCt !== null ? await env.createTime(pid) : null

  return identityStateSync(pid, recordedCt, startedAt, {
    ourPid: env.ourPid,
    ourCt: () => ourCt,
    isAlive: () => alive,
    createTime: () => actual,
    nowS: env.nowS
  })
}

export function isLiveIdentity(state: IdentityState | null): boolean {
  return state === 'ours' || state === 'match' || state === 'unknown'
}

/** Judge marker text (corpus `contract.liveness`). */
export async function judgeMarkerText(text: string, env: JudgeEnv): Promise<MarkerJudgement> {
  const marker = parseUpdateMarker(text)

  if (!marker) {
    return { verdict: 'malformed', owner: null, run: null, marker: null, ownerState: 'dead', delegateState: null }
  }

  const ownerState = await identityState(marker.pid, marker.ct, marker.startedAt, env)

  const delegateState = marker.delegate
    ? await identityState(marker.delegate.pid, marker.delegate.ct, marker.startedAt, env)
    : null

  const ownerLive = isLiveIdentity(ownerState)
  const delegateLive = isLiveIdentity(delegateState)
  const owner = ownerLive ? marker.pid : delegateLive ? marker.delegate!.pid : null
  const ours = ownerState === 'ours' || delegateState === 'ours'

  return {
    verdict: ours ? 'ours' : owner !== null ? 'live' : 'dead',
    owner,
    run: marker.run,
    marker,
    ownerState,
    delegateState
  }
}

/**
 * Whether some LIVE identity in the marker is not this process — an update
 * someone else is running. `ours` alone (a stale bridge of this very process)
 * is not an update in progress.
 */
export function hasForeignLiveIdentity(judgement: Pick<MarkerJudgement, 'ownerState' | 'delegateState'>): boolean {
  const foreign = (state: IdentityState | null) => state === 'match' || state === 'unknown'

  return foreign(judgement.ownerState) || foreign(judgement.delegateState)
}
