/**
 * backend-activity.ts
 *
 * "Is this pooled backend actually doing work?" — the question the idle reaper
 * and the LRU cap eviction in main.ts must answer before SIGTERM'ing a child.
 *
 * The renderer's 60s `hermes:backend:touch` keepalive only proves a renderer
 * socket is open for that profile. It says nothing about the backend itself:
 * when the user switches to another profile the pings stop, `lastActiveAt`
 * ages past the idle window, and the reaper killed a backend that was running
 * three agent turns plus an hour-long in-process `delegate_task` subagent.
 *
 * Every local backend now exposes `GET /api/activity` (same auth as
 * `/api/status`) reporting its own work in progress. Older backends 404 on it,
 * and a wedged process times out — both map to UNKNOWN, and unknown keeps
 * today's behaviour (reap on idle) so the app still reclaims backends from old
 * runtimes and hung processes.
 *
 * Dependency-free (same pattern as pool-eviction.ts / pool-stop.ts) so the
 * decision table is asserted directly instead of grepping main.ts.
 */

export interface BackendActivity {
  busy: boolean
  runningTurns: number
  activeSubagents: number
  backgroundProcesses: number
}

/**
 * A backend seen busy within this window is spared even if a later probe
 * blips (transient 404 from a restarting server, a timeout under load, or a
 * single idle sample between two turns). Two idle-reaper ticks (60s each).
 */
export const DEFAULT_BUSY_GRACE_MS = 120_000

function nonNegativeCount(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return null
  }

  return value
}

/**
 * Validate the `/api/activity` payload. Anything that does not carry the full
 * documented shape is null (= unknown): a partially recognised body must not
 * be trusted to prove idleness.
 */
export function parseBackendActivity(json: unknown): BackendActivity | null {
  if (!json || typeof json !== 'object' || Array.isArray(json)) {
    return null
  }

  const body = json as Record<string, unknown>

  if (body.ok === false || typeof body.busy !== 'boolean') {
    return null
  }

  const runningTurns = nonNegativeCount(body.running_turns)
  const activeSubagents = nonNegativeCount(body.active_subagents)
  const backgroundProcesses = nonNegativeCount(body.background_processes)

  if (runningTurns === null || activeSubagents === null || backgroundProcesses === null) {
    return null
  }

  return { busy: body.busy, runningTurns, activeSubagents, backgroundProcesses }
}

export type IdleReapReason = 'keepalive-fresh' | 'busy' | 'busy-grace' | 'idle' | 'unknown-activity'

export interface IdleReapInput {
  /** Milliseconds since the entry's last keepalive touch. */
  idleMs: number
  /** The pool idle window (poolIdleMs()). */
  idleLimitMs: number
  /** Probe result; null when the backend could not tell us (404 / timeout / network). */
  activity: BackendActivity | null
  /** How long a busy sighting protects the backend. */
  busyGraceMs: number
  now: number
  /** Last time this backend was seen busy, if ever. */
  lastBusyAt?: number | null
}

export interface IdleReapDecision {
  reap: boolean
  reason: IdleReapReason
  /** New busy stamp for the caller to record on the pool entry. */
  lastBusyAt?: number
}

/**
 * Decision table for one pool entry:
 *
 *   not past the idle window         → keep  (keepalive-fresh)
 *   idle + probe says busy           → keep  (busy; remember the stamp)
 *   idle + seen busy within grace    → keep  (busy-grace)
 *   idle + probe unknown             → reap  (unknown-activity — legacy path)
 *   idle + probe says not busy       → reap  (idle)
 */
export function decideIdleReap(input: IdleReapInput): IdleReapDecision {
  if (input.idleMs <= input.idleLimitMs) {
    return { reap: false, reason: 'keepalive-fresh' }
  }

  if (input.activity?.busy) {
    return { reap: false, reason: 'busy', lastBusyAt: input.now }
  }

  if (withinBusyGrace(input.lastBusyAt, input.now, input.busyGraceMs)) {
    return { reap: false, reason: 'busy-grace' }
  }

  if (input.activity === null) {
    return { reap: true, reason: 'unknown-activity' }
  }

  return { reap: true, reason: 'idle' }
}

export function withinBusyGrace(lastBusyAt: number | null | undefined, now: number, busyGraceMs: number): boolean {
  return typeof lastBusyAt === 'number' && now - lastBusyAt < busyGraceMs
}

/** The pool-entry fields the probe needs; descriptor entries have process === null. */
export interface ActivityProbeEntry {
  process?: unknown
  port?: number | null
  token?: string | null
}

/** Shape of main.ts fetchJson(url, token, { timeoutMs }). */
export type ActivityFetch = (url: string, token: string, options: { timeoutMs: number }) => Promise<unknown>

/**
 * Ask a spawned local backend for its activity. Never throws: a missing
 * process/port/token, a 404 from an older runtime, a timeout, a network error
 * or an unrecognised body all resolve to null (unknown).
 */
export async function probeBackendActivity(
  fetch: ActivityFetch,
  entry: ActivityProbeEntry,
  timeoutMs: number
): Promise<BackendActivity | null> {
  if (!entry.process || !entry.port || !entry.token) {
    return null
  }

  try {
    const body = await fetch(`http://127.0.0.1:${entry.port}/api/activity`, entry.token, { timeoutMs })

    return parseBackendActivity(body)
  } catch {
    return null
  }
}

export function describeBackendActivity(activity: BackendActivity): string {
  return (
    `running_turns=${activity.runningTurns}, active_subagents=${activity.activeSubagents}, ` +
    `background_processes=${activity.backgroundProcesses}`
  )
}

/** desktop.log line for a backend the idle reaper left alone because it was busy. */
export function formatSparedBusyBackendLog(key: string, activity: BackendActivity): string {
  return `Sparing busy profile backend "${key}" (${describeBackendActivity(activity)}) despite idle keepalive`
}

/** desktop.log line for a backend spared by the LRU cap because it was busy. */
export function formatSparedBusyEvictionLog(key: string, activity: BackendActivity, cap: number): string {
  return `Sparing busy profile backend "${key}" (${describeBackendActivity(activity)}) from LRU cap ${cap}`
}

/** desktop.log line for a backend spared because it was seen busy within the grace window. */
export function formatSparedRecentlyBusyLog(key: string, lastBusyAt: number, now: number, busyGraceMs: number): string {
  return (
    `Sparing recently busy profile backend "${key}" ` +
    `(busy ${Math.round((now - lastBusyAt) / 1000)}s ago, within ${Math.round(busyGraceMs / 1000)}s grace)`
  )
}
