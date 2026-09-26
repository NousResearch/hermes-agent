import type { StatusTone } from '@/components/status-dot'
import type { CronJob } from '@/types/hermes'

// Semantic tone per cron job state. The shared StatusDot turns these into
// shape + color so sidebar and overlay never rely on hue alone.
export const STATE_TONE: Record<string, StatusTone> = {
  completed: 'muted',
  disabled: 'muted',
  enabled: 'good',
  error: 'bad',
  paused: 'warn',
  running: 'good',
  scheduled: 'good'
}

// Effective state: explicit state wins; otherwise infer from the enabled flag.
export function jobState(job: CronJob): string {
  const state = typeof job.state === 'string' ? job.state.trim() : ''

  return state || (job.enabled === false ? 'disabled' : 'scheduled')
}

// Human label for a job: name → first 60 of prompt → first 60 of script → id.
// One source for the sidebar row and the Cron page so the two never drift.
export function jobTitle(job: CronJob): string {
  const pick = (v: unknown) => (typeof v === 'string' ? v.trim() : '')
  const clip = (v: string) => (v.length > 60 ? `${v.slice(0, 60)}…` : v)

  return pick(job.name) || clip(pick(job.prompt)) || clip(pick(job.script)) || job.id || 'Cron job'
}

// Mirrors hermes_cli/cron.py `_OVERDUE_GRACE_SECONDS`: a busy tick can dispatch a few minutes late.
export const NEXT_RUN_OVERDUE_GRACE_MS = 15 * 60 * 1000

// Milliseconds a job's stored next_run_at has sat in the past beyond that grace, or null when
// the slot is upcoming, within grace, unparseable, or the job is not expected to fire. A slot
// parked in the past is the only user-visible trace of a dead scheduler (#114309), so no
// surface may present it as an upcoming "Next".
export function nextRunOverdueMs(
  job: { enabled?: boolean; next_run_at?: null | string; state?: null | string },
  nowMs = Date.now()
): null | number {
  const state = jobState(job as CronJob)

  if (state === 'paused' || state === 'completed' || state === 'disabled' || !job.next_run_at) {
    return null
  }

  const at = Date.parse(job.next_run_at)

  if (Number.isNaN(at)) {
    return null
  }

  const overdue = nowMs - at

  return overdue > NEXT_RUN_OVERDUE_GRACE_MS ? overdue : null
}
