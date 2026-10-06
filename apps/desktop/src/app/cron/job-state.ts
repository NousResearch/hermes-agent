import type { CronJob } from '@/types/hermes'

// Status-pip color per cron job state. Single source for the sidebar section and
// the Cron page so the two never drift. (Animation/size live at the call site.)
export const STATE_DOT: Record<string, string> = {
  completed: 'bg-(--ui-text-quaternary)',
  disabled: 'bg-(--ui-text-quaternary)',
  enabled: 'bg-primary',
  error: 'bg-destructive',
  paused: 'bg-amber-500',
  running: 'bg-primary',
  scheduled: 'bg-primary'
}

// Effective state: explicit state wins; otherwise infer from the enabled flag.
export function jobState(job: CronJob): string {
  const state = typeof job.state === 'string' ? job.state.trim() : ''

  return state || (job.enabled === false ? 'disabled' : 'scheduled')
}

// Grapheme segmentation, the same idiom as profile-short-label.ts: code points alone still
// cut ZWJ sequences, flags, skin tones, and combining marks mid-cluster. A job name is raw
// user input, so "🇨🇳 每日简报" is a plausible value to keep intact.
const truncateGraphemes = new Intl.Segmenter(undefined, { granularity: 'grapheme' })

export function truncateText(value: string, max: number): string {
  const graphemes = Array.from(truncateGraphemes.segment(value), ({ segment }) => segment)

  return graphemes.length > max ? `${graphemes.slice(0, max).join('')}…` : value
}

// Human label for a job: name → first 60 of prompt → first 60 of script → id.
// One source for the sidebar row and the Cron page so the two never drift.
export function jobTitle(job: CronJob): string {
  const pick = (v: unknown) => (typeof v === 'string' ? v.trim() : '')

  return pick(job.name) || truncateText(pick(job.prompt), 60) || truncateText(pick(job.script), 60) || job.id || 'Cron job'
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
