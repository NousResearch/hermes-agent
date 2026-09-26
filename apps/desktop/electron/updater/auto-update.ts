// updater/auto-update.ts — opt-in "install updates on the first launch after
// login" policy (#123674).
//
// Electron owns this decision because every input is a machine fact: which OS
// login session we are in, whether a messaging gateway is mid-turn, and what
// this device already attempted. The renderer only asks "may I run the
// automatic update now?" and reports how it went; the apply itself reuses the
// ordinary installation-specific update flow (checkout hand-off, native mac
// updater, …) so there is no second pull/install/restart implementation.
//
// Policy, in order:
//   1. Off unless the user turned it on (device-scoped, userData JSON).
//   2. macOS and Linux only for now.
//   3. At most ONE automatic attempt per OS login session. The session is
//      claimed before the attempt starts, so the relaunch that follows an
//      update — or a crash mid-update — never re-triggers it, and a failed
//      update is not retried on every relaunch.
//   4. Never interrupt work: while any gateway reports in-flight agent turns
//      (or this Desktop has turns running) the claim is deferred, not
//      consumed. Deferral is bounded; past the budget the session is given up.
//
// Pure data in, decisions out. The impure edges (ps, /proc, fs, pid liveness)
// are injected so the policy is testable without a real login session.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

export const AUTO_UPDATE_STATE_VERSION = 1

/** Poll cadence while deferring behind busy gateways. */
export const AUTO_UPDATE_DEFER_RETRY_MS = 60_000

/** Give up on this login session after deferring this long. */
export const AUTO_UPDATE_MAX_DEFER_MS = 30 * 60_000

/** gateway_state.json heartbeat is re-stamped every 60s; older = not live (gateway/status.py). */
export const GATEWAY_STATUS_STALE_MS = 120_000

export type AutoUpdateOutcome =
  | 'handed-off'
  | 'updated'
  | 'up-to-date'
  | 'failed'
  | 'skipped-dirty'
  | 'skipped-unsupported'
  | 'check-failed'
  | 'deferred-timeout'

export interface AutoUpdateAttempt {
  sessionKey: string
  at: number
  outcome: AutoUpdateOutcome
  target?: string
  message?: string
}

export interface AutoUpdateState {
  version: number
  enabled: boolean
  /** The login session whose one automatic attempt has been used. */
  claimedSessionKey?: string
  lastAttempt?: AutoUpdateAttempt
  /** When deferral for the current session began (busy gateway). */
  deferral?: { sessionKey: string; since: number }
}

export const DEFAULT_AUTO_UPDATE_STATE: AutoUpdateState = { version: AUTO_UPDATE_STATE_VERSION, enabled: false }

const OUTCOMES: ReadonlySet<string> = new Set<AutoUpdateOutcome>([
  'handed-off',
  'updated',
  'up-to-date',
  'failed',
  'skipped-dirty',
  'skipped-unsupported',
  'check-failed',
  'deferred-timeout'
])

const str = (value: unknown): string | undefined => (typeof value === 'string' && value.length > 0 ? value : undefined)

const num = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined

/** Coerce whatever is on disk (hand edits, older/newer builds) into a valid state. */
export function parseAutoUpdateState(raw: unknown): AutoUpdateState {
  if (!raw || typeof raw !== 'object') {
    return { ...DEFAULT_AUTO_UPDATE_STATE }
  }

  const record = raw as Record<string, unknown>
  const state: AutoUpdateState = { version: AUTO_UPDATE_STATE_VERSION, enabled: record.enabled === true }
  const claimed = str(record.claimedSessionKey)

  if (claimed) {
    state.claimedSessionKey = claimed
  }

  const attempt = record.lastAttempt as Record<string, unknown> | undefined
  const attemptKey = str(attempt?.sessionKey)
  const attemptAt = num(attempt?.at)
  const outcome = str(attempt?.outcome)

  if (attemptKey && attemptAt !== undefined && outcome && OUTCOMES.has(outcome)) {
    state.lastAttempt = {
      sessionKey: attemptKey,
      at: attemptAt,
      outcome: outcome as AutoUpdateOutcome,
      ...(str(attempt?.target) ? { target: str(attempt?.target) } : {}),
      ...(str(attempt?.message) ? { message: str(attempt?.message)!.slice(0, 500) } : {})
    }
  }

  const deferral = record.deferral as Record<string, unknown> | undefined
  const deferralKey = str(deferral?.sessionKey)
  const deferralSince = num(deferral?.since)

  if (deferralKey && deferralSince !== undefined) {
    state.deferral = { sessionKey: deferralKey, since: deferralSince }
  }

  return state
}

// ── Login-session identity ──────────────────────────────────────────────────

export interface LoginSessionDeps {
  platform: NodeJS.Platform
  uid: number
  env: NodeJS.ProcessEnv
  /** Run a command, return stdout; throw on failure. Must be bounded. */
  run: (command: string, args: string[]) => string
  readFile: (file: string) => string
  uptimeSeconds: () => number
  now: () => number
}

/**
 * A stable key for the current OS login session: identical for every launch
 * inside one login, different after logout/login or reboot.
 *
 * - macOS: each GUI login gets its own `loginwindow` process owned by the
 *   user; its pid + start time names the session (fast user switching keeps a
 *   separate one per user).
 * - Linux: systemd-logind's XDG_SESSION_ID scoped by the kernel boot id (the
 *   session counter restarts at boot).
 * - Anything unreadable falls back to the boot time, i.e. "first launch after
 *   boot" — the closest safe approximation, never "every launch".
 */
export function resolveLoginSessionKey(deps: LoginSessionDeps): string {
  if (deps.platform === 'darwin') {
    try {
      const pids = deps
        .run('/usr/bin/pgrep', ['-u', String(deps.uid), '-x', 'loginwindow'])
        .split(/\s+/)
        .filter(token => /^\d+$/.test(token))

      const pid = pids[0]

      if (pid) {
        const started = deps.run('/bin/ps', ['-o', 'lstart=', '-p', pid]).trim().replace(/\s+/g, ' ')

        if (started) {
          return `darwin:loginwindow:${pid}:${started}`
        }
      }
    } catch {
      // fall through to boot time
    }

    try {
      const boot = /sec\s*=\s*(\d+)/.exec(deps.run('/usr/sbin/sysctl', ['-n', 'kern.boottime']))?.[1]

      if (boot) {
        return `darwin:boot:${boot}`
      }
    } catch {
      // fall through
    }
  }

  if (deps.platform === 'linux') {
    let bootId = ''

    try {
      bootId = deps.readFile('/proc/sys/kernel/random/boot_id').trim()
    } catch {
      bootId = ''
    }

    const sessionId = (deps.env.XDG_SESSION_ID || '').trim()

    if (bootId && sessionId) {
      return `linux:${bootId}:session:${sessionId}`
    }

    if (bootId) {
      return `linux:${bootId}`
    }
  }

  // Five-minute buckets absorb uptime jitter between launches.
  const bootEpochSeconds = deps.now() / 1000 - deps.uptimeSeconds()

  return `boot:${Math.floor(bootEpochSeconds / 300)}`
}

// ── Gateway activity ────────────────────────────────────────────────────────

export interface GatewayActivity {
  busy: boolean
  activeAgents: number
  /** Home directories whose gateway reported in-flight work. */
  busyHomes: string[]
}

const LIVE_GATEWAY_STATES: ReadonlySet<string> = new Set(['running', 'degraded', 'draining', 'starting'])

/**
 * Whether one `gateway_state.json` record describes a live gateway with
 * in-flight work. A dead pid or a stale heartbeat is not a blocker — that is a
 * leftover file, and deferring on it would wedge automatic updates forever.
 */
export function gatewayRecordBusy(record: unknown, now: number, isPidAlive: (pid: number) => boolean): number {
  if (!record || typeof record !== 'object') {
    return 0
  }

  const raw = record as Record<string, unknown>
  const state = str(raw.gateway_state)

  if (!state || !LIVE_GATEWAY_STATES.has(state)) {
    return 0
  }

  const pid = num(raw.pid)

  if (pid === undefined || pid <= 0 || !isPidAlive(pid)) {
    return 0
  }

  const updatedAt = Date.parse(String(raw.updated_at ?? ''))

  if (!Number.isFinite(updatedAt) || now - updatedAt > GATEWAY_STATUS_STALE_MS) {
    return 0
  }

  const active = Number(raw.active_agents)

  return Number.isFinite(active) && active > 0 ? Math.floor(active) : 0
}

export interface GatewayScanDeps {
  hermesHome: string
  readFile: (file: string) => string
  listDir: (dir: string) => string[]
  isPidAlive: (pid: number) => boolean
  now: () => number
}

/** The root home plus every named profile — `hermes update` restarts them all. */
export function scanGatewayActivity(deps: GatewayScanDeps): GatewayActivity {
  const homes: string[] = [deps.hermesHome]

  try {
    for (const name of deps.listDir(path.join(deps.hermesHome, 'profiles'))) {
      homes.push(path.join(deps.hermesHome, 'profiles', name))
    }
  } catch {
    // no profiles dir
  }

  let activeAgents = 0
  const busyHomes: string[] = []

  for (const home of homes) {
    let record: unknown = null

    try {
      record = JSON.parse(deps.readFile(path.join(home, 'gateway_state.json')))
    } catch {
      continue
    }

    const active = gatewayRecordBusy(record, deps.now(), deps.isPidAlive)

    if (active > 0) {
      activeAgents += active
      busyHomes.push(home)
    }
  }

  return { busy: activeAgents > 0, activeAgents, busyHomes }
}

// ── The claim ───────────────────────────────────────────────────────────────

export type AutoUpdateClaimReason =
  | 'disabled'
  | 'unsupported-platform'
  | 'already-ran-this-session'
  | 'busy'
  | 'deferred-too-long'
  | 'first-launch-after-login'

export interface AutoUpdateClaim {
  action: 'run' | 'skip' | 'defer'
  reason: AutoUpdateClaimReason
  sessionKey: string
  /** For `defer`: when to ask again. */
  retryInMs?: number
  activeAgents?: number
}

export interface ClaimInput {
  state: AutoUpdateState
  sessionKey: string
  platform: NodeJS.Platform
  gatewayActiveAgents: number
  desktopActiveTurns: number
  now: number
  maxDeferMs?: number
  retryMs?: number
}

export function autoUpdatePlatformSupported(platform: NodeJS.Platform): boolean {
  return platform === 'darwin' || platform === 'linux'
}

/**
 * Decide whether this launch may run the automatic update. Returns the next
 * state to persist alongside the decision: a `run` consumes the session up
 * front (so a crash or the post-update relaunch never repeats it); a `defer`
 * only records when waiting began; a deferral past budget consumes the
 * session with a `deferred-timeout` attempt.
 */
export function decideAutoUpdateClaim(input: ClaimInput): { claim: AutoUpdateClaim; nextState: AutoUpdateState } {
  const { state, sessionKey, now } = input
  const maxDeferMs = input.maxDeferMs ?? AUTO_UPDATE_MAX_DEFER_MS
  const retryMs = input.retryMs ?? AUTO_UPDATE_DEFER_RETRY_MS
  const base = { sessionKey }

  if (!state.enabled) {
    return { claim: { ...base, action: 'skip', reason: 'disabled' }, nextState: state }
  }

  if (!autoUpdatePlatformSupported(input.platform)) {
    return { claim: { ...base, action: 'skip', reason: 'unsupported-platform' }, nextState: state }
  }

  if (state.claimedSessionKey === sessionKey) {
    return { claim: { ...base, action: 'skip', reason: 'already-ran-this-session' }, nextState: state }
  }

  const activeAgents = Math.max(0, input.gatewayActiveAgents) + Math.max(0, input.desktopActiveTurns)

  if (activeAgents > 0) {
    const since = state.deferral?.sessionKey === sessionKey ? state.deferral.since : now

    if (now - since >= maxDeferMs) {
      return {
        claim: { ...base, action: 'skip', reason: 'deferred-too-long', activeAgents },
        nextState: {
          ...state,
          claimedSessionKey: sessionKey,
          deferral: undefined,
          lastAttempt: { sessionKey, at: now, outcome: 'deferred-timeout' }
        }
      }
    }

    return {
      claim: { ...base, action: 'defer', reason: 'busy', retryInMs: retryMs, activeAgents },
      nextState: { ...state, deferral: { sessionKey, since } }
    }
  }

  return {
    claim: { ...base, action: 'run', reason: 'first-launch-after-login' },
    nextState: { ...state, claimedSessionKey: sessionKey, deferral: undefined }
  }
}

/** Record how the claimed attempt ended (shown in the UI, never re-tried this session). */
export function recordAutoUpdateOutcome(
  state: AutoUpdateState,
  report: { sessionKey: string; outcome: AutoUpdateOutcome; target?: string; message?: string },
  now: number
): AutoUpdateState {
  if (!OUTCOMES.has(report.outcome)) {
    return state
  }

  return {
    ...state,
    claimedSessionKey: state.claimedSessionKey ?? report.sessionKey,
    lastAttempt: {
      sessionKey: report.sessionKey,
      at: now,
      outcome: report.outcome,
      ...(report.target ? { target: report.target.slice(0, 200) } : {}),
      ...(report.message ? { message: report.message.slice(0, 500) } : {})
    }
  }
}

// ── Production edges ────────────────────────────────────────────────────────

export function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)

    return true
  } catch (error) {
    // EPERM: exists, owned by someone else — still alive.
    return (error as NodeJS.ErrnoException)?.code === 'EPERM'
  }
}

export function defaultLoginSessionDeps(run: LoginSessionDeps['run']): LoginSessionDeps {
  return {
    platform: process.platform,
    uid: typeof process.getuid === 'function' ? process.getuid() : -1,
    env: process.env,
    run,
    readFile: file => fs.readFileSync(file, 'utf8'),
    uptimeSeconds: () => os.uptime(),
    now: () => Date.now()
  }
}

export function defaultGatewayScanDeps(hermesHome: string): GatewayScanDeps {
  return {
    hermesHome,
    readFile: file => fs.readFileSync(file, 'utf8'),
    listDir: dir =>
      fs
        .readdirSync(dir, { withFileTypes: true })
        .filter(entry => entry.isDirectory())
        .map(entry => entry.name),
    isPidAlive,
    now: () => Date.now()
  }
}

export function readAutoUpdateState(file: string): AutoUpdateState {
  try {
    return parseAutoUpdateState(JSON.parse(fs.readFileSync(file, 'utf8')))
  } catch {
    return { ...DEFAULT_AUTO_UPDATE_STATE }
  }
}

export function writeAutoUpdateState(file: string, state: AutoUpdateState): void {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  const tmp = `${file}.tmp`
  fs.writeFileSync(tmp, JSON.stringify(state, null, 2))
  fs.renameSync(tmp, file)
}
