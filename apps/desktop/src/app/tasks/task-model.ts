import { chatMessageText, type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import { latestSessionTodos, type TodoItem } from '@/lib/todos'
import type { CronJob, SessionInfo, SessionMessage } from '@/types/hermes'

import { jobState, jobTitle } from '../cron/job-state'

// ── What a "task" is here (honest scope) ────────────────────────────────────
// hc-419 asks for a Codex-style "Goal mode": hand off one big job, let it run in
// the background, watch progress, get notified on completion. The runtime has NO
// durable multi-day autonomous executor — `delegate_task(background=true)` runs
// in-process daemon threads that die on session close / process restart (see
// WORK-NOTES-hc419-goal-mode.md). The one real, persistent background executor
// is the cron scheduler, which natively supports **one-shot** jobs
// (`schedule.kind === 'once'`, e.g. "2h" / a timestamp). Those persist to
// jobs.json, run in the backend, produce a real session, and survive an app
// restart. So a "task" == a one-shot cron job. Recurring cron jobs stay on the
// /cron page; this page shows only the one-shot long-runs.

/** Coarse lifecycle bucket a one-shot task is shown under. */
export type TaskPhase = 'running' | 'done' | 'failed'

/** How confident we are the task is stuck (running but no recent activity). */
export const STUCK_AFTER_MS = 20 * 60 * 1000 // 20 min of no run-session activity

export interface TaskProgress {
  /** Steps the agent has planned via the `todo` tool, if any. Empty when the
   *  run hasn't emitted a plan yet — we never fabricate steps. */
  todos: TodoItem[]
  completedSteps: number
  totalSteps: number
  /** The step currently in progress, if the plan marks one. */
  currentStep: null | string
  /** Most recent assistant text from the run — the "latest output" line. Null
   *  when the run has produced no assistant text yet. */
  latestOutput: null | string
}

const asText = (value: unknown): string => (typeof value === 'string' ? value : '')

/** A short, single-line label for a task, reusing the cron title rule (name →
 *  prompt head → id). Re-exported so stores/notifications don't reach into the
 *  cron page's internals directly. */
export function jobTitleShort(job: CronJob): string {
  return jobTitle(job)
}

/** True for a cron job that represents a one-shot run (our "task"), as opposed
 *  to a recurring schedule. Reads the structured `schedule.kind` first, then
 *  falls back to the display string the backend stamps ("once …"). */
export function isOneShotJob(job: CronJob): boolean {
  const kind = asText(job.schedule?.kind).trim().toLowerCase()

  if (kind) {
    return kind === 'once'
  }

  // Legacy / partial payloads without a structured kind: the backend's
  // schedule_display for one-shots always starts with "once".
  const display = (asText(job.schedule_display) || asText(job.schedule?.display)).trim().toLowerCase()

  return display.startsWith('once')
}

/** Map a job's cron state onto the task lifecycle bucket. A completed one-shot
 *  ends `enabled:false, state:"completed"`; a failed recurring-compute ends
 *  `state:"error"`. `disabled` (paused-then-expired, hand-disabled) reads as
 *  done rather than failed — nothing went wrong, it just isn't live. */
export function taskPhase(job: CronJob): TaskPhase {
  const state = jobState(job)

  if (state === 'error') {
    return 'failed'
  }

  if (state === 'completed' || state === 'disabled') {
    return 'done'
  }

  // scheduled / running / enabled / paused → still an in-flight task.
  return 'running'
}

/** The single run session that best represents a task's progress: the newest
 *  one. One-shots run at most once, but a manual "trigger now" can add a second
 *  row, so newest-by-time wins. Runs arrive newest-first from the API already;
 *  we don't rely on that and pick explicitly. */
export function primaryRun(runs: readonly SessionInfo[]): null | SessionInfo {
  let best: null | SessionInfo = null
  let bestAt = -Infinity

  for (const run of runs) {
    const at = run.last_active || run.started_at || 0

    if (at >= bestAt) {
      bestAt = at
      best = run
    }
  }

  return best
}

/** Derive progress from a run session's stored transcript. Reuses the exact
 *  converters the live chat uses (`toChatMessages` → `latestSessionTodos`) so
 *  the plan we show matches what the session view would render. Everything is
 *  real transcript data — no invented fields. */
export function deriveProgress(messages: readonly SessionMessage[]): TaskProgress {
  const chat: ChatMessage[] = toChatMessages(messages as SessionMessage[])
  const todos = latestSessionTodos(chat) ?? []

  const completedSteps = todos.filter(t => t.status === 'completed').length
  const currentStep = todos.find(t => t.status === 'in_progress')?.content ?? null

  return {
    todos,
    completedSteps,
    totalSteps: todos.length,
    currentStep,
    latestOutput: latestAssistantText(chat)
  }
}

/** Last non-empty assistant text across the transcript — the freshest thing the
 *  agent "said". Skips tool-only assistant turns. */
function latestAssistantText(chat: readonly ChatMessage[]): null | string {
  for (let i = chat.length - 1; i >= 0; i -= 1) {
    const message = chat[i]

    if (message?.role !== 'assistant') {
      continue
    }

    const text = chatMessageText(message).trim()

    if (text) {
      return text
    }
  }

  return null
}

/** A running task whose run session hasn't advanced in a long time is likely
 *  stuck (backend killed mid-turn, model wedged, etc.). We flag it rather than
 *  claim it's still working. `now` is injected for testability. */
export function isStuck(job: CronJob, run: null | SessionInfo, now: number): boolean {
  if (taskPhase(job) !== 'running') {
    return false
  }

  // A run that reports itself active is, by definition, not stuck.
  if (run?.is_active) {
    return false
  }

  // No run yet = queued/waiting for its scheduled time, not stuck.
  if (!run) {
    return false
  }

  const lastActiveMs = (run.last_active || run.started_at || 0) * 1000

  if (!lastActiveMs) {
    return false
  }

  return now - lastActiveMs > STUCK_AFTER_MS
}

/** Build the one-shot "schedule" string the cron backend understands from the
 *  editor's when-choice. "now" maps to a 1-minute delay so the ticker (60s
 *  loop) picks it up on its next pass without racing the create call. */
export function scheduleStringForWhen(when: TaskWhen): string {
  switch (when.kind) {
    case 'now':
      // The scheduler ticks every 60s; "1m" makes it due on the next tick.
      return '1m'
    case 'in':
      return when.value.trim()
    case 'at':
      return when.value.trim()
  }
}

export type TaskWhen =
  | { kind: 'now' }
  /** Relative duration string the backend parses: "30m", "2h", "1d". */
  | { kind: 'in'; value: string }
  /** ISO-ish timestamp the backend parses: "2026-02-03T14:00". */
  | { kind: 'at'; value: string }
