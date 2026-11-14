import type { CronJob, SessionInfo } from '@/hermes'

import { parseCronDeliveryTargets } from '../cron/cron-job-model'
import { jobDeliveryFailed, jobLastRunFailed, jobState, jobTitle, nextRunOverdueMs } from '../cron/job-state'
import { jobRunInFlight } from '../cron/run-control'

// What the Routines page derives from a job record: its group in the panel, the status chip, the
// reason it needs attention, where its results go, and the outcome chip of each run. Pure, so the
// rules are tested as data (routine-model.test.ts).

/** Why a routine needs attention (proposal point 3: "Problems first"). */
export type RoutineIssue = 'delivery' | 'failed' | 'overdue'

export type RoutineGroup = 'attention' | 'finished' | 'paused' | 'scheduled'

export const ROUTINE_GROUPS: readonly RoutineGroup[] = ['attention', 'scheduled', 'paused', 'finished']

export type RoutineStatus = 'delivery' | 'failed' | 'finished' | 'on' | 'overdue' | 'paused' | 'running'

export type RoutineFilter = 'all' | 'attention' | 'paused' | 'scheduled'

export const ROUTINE_FILTERS: readonly RoutineFilter[] = ['all', 'attention', 'scheduled', 'paused']

const RESTING = new Set(['completed', 'disabled', 'paused'])

/** A routine that is not expected to fire again on its own: paused, switched off, or done. */
function resting(job: CronJob): boolean {
  return RESTING.has(jobState(job))
}

/**
 * The problem a routine has, if any: its newest run failed, ran but did not reach its destination, or
 * a scheduled slot sat in the past beyond the scheduler's grace (no scheduler is ticking). A run in
 * flight supersedes the previous result, as it does for the chat sidebar's attention.
 */
export function routineIssue(job: CronJob, nowMs = Date.now()): null | RoutineIssue {
  if (jobRunInFlight(job)) {
    return null
  }

  if (jobLastRunFailed(job)) {
    return 'failed'
  }

  if (jobDeliveryFailed(job)) {
    return 'delivery'
  }

  return nextRunOverdueMs(job, nowMs) === null ? null : 'overdue'
}

export function routineGroup(job: CronJob, nowMs = Date.now()): RoutineGroup {
  if (routineIssue(job, nowMs)) {
    return 'attention'
  }

  const state = jobState(job)

  if (state === 'completed') {
    return 'finished'
  }

  return state === 'paused' || state === 'disabled' ? 'paused' : 'scheduled'
}

export function routineNeedsAttention(job: CronJob, nowMs = Date.now()): boolean {
  return routineIssue(job, nowMs) !== null
}

/** The panel's groups, each sorted by title; a group with nothing in it is left out by the caller. */
export function groupRoutines(jobs: readonly CronJob[], nowMs = Date.now()): Record<RoutineGroup, CronJob[]> {
  const groups: Record<RoutineGroup, CronJob[]> = { attention: [], finished: [], paused: [], scheduled: [] }

  for (const job of jobs) {
    groups[routineGroup(job, nowMs)].push(job)
  }

  for (const group of ROUTINE_GROUPS) {
    groups[group].sort(compareRoutines)
  }

  return groups
}

export function compareRoutines(a: CronJob, b: CronJob): number {
  return jobTitle(a).localeCompare(jobTitle(b)) || (a.profile ?? '').localeCompare(b.profile ?? '')
}

/** The one chip a routine wears: a live run first, then a problem, then its schedule state. */
export function routineStatus(job: CronJob, live: boolean, nowMs = Date.now()): RoutineStatus {
  if (live) {
    return 'running'
  }

  const issue = routineIssue(job, nowMs)

  if (issue) {
    return issue
  }

  const state = jobState(job)

  return state === 'completed' ? 'finished' : resting(job) ? 'paused' : 'on'
}

/** Chip tone by status: problems in the danger family, a live run in the brand selection. */
export const STATUS_TONE: Record<RoutineStatus, 'bad' | 'good' | 'muted' | 'running' | 'warn'> = {
  delivery: 'warn',
  failed: 'bad',
  finished: 'muted',
  on: 'good',
  overdue: 'warn',
  paused: 'muted',
  running: 'running'
}

const FILTER_GROUPS: Record<RoutineFilter, null | RoutineGroup> = {
  all: null,
  attention: 'attention',
  paused: 'paused',
  scheduled: 'scheduled'
}

/** The overview's rows: the segment's group, then every word of the query in the title, instruction,
 *  agent or schedule. */
export function filterRoutines(
  jobs: readonly CronJob[],
  filter: RoutineFilter,
  query: string,
  agentLabel: (profile: string) => string,
  nowMs = Date.now()
): CronJob[] {
  const group = FILTER_GROUPS[filter]
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean)

  return jobs
    .filter(job => !group || routineGroup(job, nowMs) === group)
    .filter(job => {
      if (!words.length) {
        return true
      }

      const haystack = [jobTitle(job), job.prompt ?? '', agentLabel(job.profile ?? ''), job.schedule_display ?? '']
        .join(' ')
        .toLowerCase()

      return words.every(word => haystack.includes(word))
    })
    .sort(compareRoutines)
}

/** `[bot:name] ` is the Bots plugin's routine-name namespace, not part of what the user called it. */
const BOT_NAMESPACE = /^\[bot:[a-z0-9][a-z0-9_-]*\]\s*/i

export function routineTitle(job: CronJob): string {
  const title = jobTitle(job)

  return title.replace(BOT_NAMESPACE, '') || title
}

// ── Delivery ──────────────────────────────────────────────────────────────────────────────────────

/** Where a routine sends its results, as target ids. `local` is "save only" (every run is kept in
 *  Runs anyway), so it is the empty selection, not a destination. */
export function deliveryTargets(deliver: null | string | undefined): string[] {
  return parseCronDeliveryTargets(deliver ?? '').filter(target => target !== 'local')
}

/** Toggle one destination; nothing left selected is `local` (save only), never an empty string. */
export function toggleDelivery(deliver: string, target: string, checked: boolean): string {
  const current = deliveryTargets(deliver)
  const next = checked ? [...new Set([...current, target])] : current.filter(candidate => candidate !== target)

  return next.length ? next.join(',') : 'local'
}

/** A delivery target that names the routine's agent's own chat (`bot-chat` or `bot-chat:<profile>`). */
export function agentChatTarget(target: string): null | string {
  if (target === 'bot-chat') {
    return ''
  }

  return target.startsWith('bot-chat:') ? target.slice('bot-chat:'.length) : null
}

// ── Advanced fields ───────────────────────────────────────────────────────────────────────────────

const SELF = 'self'

/** "Remember last run": the job's own previous output is injected into each run (`context_from: self`). */
export function remembersLastRun(job: Pick<CronJob, 'context_from'> | null | undefined): boolean {
  return Boolean(job?.context_from?.some(ref => ref.trim().toLowerCase() === SELF))
}

/** The job's context refs with `self` added or removed, keeping any chain to another job. */
export function withRememberLastRun(refs: null | readonly string[] | undefined, on: boolean): string[] {
  const others = (refs ?? []).filter(ref => ref.trim().toLowerCase() !== SELF)

  return on ? [...others, SELF] : others
}

export function runLimit(job: Pick<CronJob, 'repeat'> | null | undefined): null | number {
  const times = job?.repeat?.times

  return typeof times === 'number' && times > 0 ? times : null
}

// ── Runs ──────────────────────────────────────────────────────────────────────────────────────────

export type RunChip = 'delivery' | 'failed' | 'running' | 'stopped'

const EXECUTION_CHIP: Record<string, RunChip> = {
  cancelled: 'stopped',
  claimed: 'running',
  failed: 'failed',
  running: 'running'
}

/**
 * A run's outcome chip, only where the backend says so: a run the scheduler still owns is running; the
 * run the job's newest ledger attempt names carries that attempt's verdict. Older runs have no
 * recorded outcome on the row, so they carry none (never a guessed "Succeeded").
 */
export function runChip(run: SessionInfo, job: Pick<CronJob, 'latest_execution'>): null | RunChip {
  if (run.source === 'cron' && run.ended_at == null && run.scheduler_owned) {
    return 'running'
  }

  const output = outputRunStatus(run)

  if (output) {
    return output.chip
  }

  const execution = job.latest_execution

  if (!execution || !run.cron_execution_id || run.cron_execution_id !== execution.id) {
    return null
  }

  return EXECUTION_CHIP[execution.status] ?? (execution.delivery_outcome === 'failed' ? 'delivery' : null)
}

/** A run with no session behind it (a script-only fire's output, a ledger row) is titled by the backend
 *  as `<STATUS> · <reason>` (`_execution_status_title`); read its status and reason back from it. */
const OUTPUT_TITLE = /^([A-Z][A-Z ]*[A-Z]) · ([\s\S]*)$/

const OUTPUT_CHIP: Record<string, RunChip> = {
  CANCELLED: 'stopped',
  'DELIVERY FAILED': 'delivery',
  FAILED: 'failed'
}

export function outputRunStatus(run: Pick<SessionInfo, 'source' | 'title'>): { chip: null | RunChip; reason: string } | null {
  const match = run.source === 'cron' ? null : OUTPUT_TITLE.exec(run.title ?? '')

  return match ? { chip: OUTPUT_CHIP[match[1]] ?? null, reason: match[2] } : null
}

/** Seconds a closed run took, or null when the row cannot say. */
export function runDurationSeconds(run: Pick<SessionInfo, 'ended_at' | 'source' | 'started_at'>): null | number {
  if (run.source !== 'cron' || run.ended_at == null || !run.started_at) {
    return null
  }

  const seconds = run.ended_at - run.started_at

  return seconds >= 0 ? seconds : null
}
