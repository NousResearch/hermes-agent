import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import type { CronJob } from '@/hermes'
import { useI18n } from '@/i18n'
import { $profiles, profileLabel } from '@/store/profile'

import { plainRunError, routineErrorText } from './routine-error'
import { type RoutineIssue, routineIssue, routinePaused, type RoutineStatus, STATUS_TONE } from './routine-model'
import { capitalized, formatMoment, runDay } from './routine-schedule'
import { type RoutinesCopy, useRoutinesCopy } from './routines-copy'

/** An agent's display name by profile key, from this machine's profile list. */
export function useAgentLabel(): (profile: string) => string {
  const profiles = useStore($profiles)

  return profile => {
    const found = profiles.find(candidate => candidate.name === profile)

    return found ? profileLabel(found) : profile || 'default'
  }
}

/** The status chip (9px uppercase, Figma 1098:8495): problems in the danger family, a live run in the
 *  brand selection, everything else neutral with its status ink. */
export function StatusChip({ status }: { status: RoutineStatus }) {
  const copy = useRoutinesCopy()

  return (
    <span className="routines-chip" data-tone={STATUS_TONE[status]}>
      {copy.status[status]}
    </span>
  )
}

/** The agent a routine belongs to (proposal: rows never said which agent). */
export function AgentTag({ label }: { label: string }) {
  return (
    <span className="routines-agent-tag" title={label}>
      {label}
    </span>
  )
}

/** The banner's heading: which run went wrong, by its day ("Yesterday’s run failed"). */
function bannerTitle(issue: RoutineIssue, job: CronJob, copy: RoutinesCopy, locale: string): string {
  if (issue === 'overdue') {
    return copy.banner.overdue
  }

  const streak = job.failure_streak ?? 0

  if (issue === 'failed' && streak > 1) {
    return copy.banner.failedStreak(streak)
  }

  const day = job.last_run_at ? runDay(job.last_run_at, locale) : null
  const words = issue === 'failed' ? copy.banner.failed : copy.banner.notDelivered

  return day ? words[day.kind](day.date) : words.last
}

interface RoutineBannerProps {
  job: CronJob
  busy: boolean
  onRetry: () => void
}

/** A routine whose last run went wrong (rev 1383:21029): which run, the chip, the reason in one plain
 *  sentence, when it tries again, and Retry. Shown only while there is a problem. */
export function RoutineBanner({ busy, job, onRetry }: RoutineBannerProps) {
  const copy = useRoutinesCopy()
  const agentLabel = useAgentLabel()
  const { locale } = useI18n()
  const issue = routineIssue(job)

  if (!issue) {
    return null
  }

  const next =
    issue === 'overdue'
      ? copy.banner.overdueHint
      : job.next_run_at && !routinePaused(job)
        ? copy.banner.retryAt(formatMoment(job.next_run_at, locale))
        : ''

  const raw = routineErrorText(job)
  const detail = raw ? plainRunError(raw, copy, agentLabel(job.profile ?? '')) : ''

  return (
    <div className="routines-banner" role="status">
      <div className="routines-banner-text">
        <p className="routines-banner-title">
          <span>{capitalized(bannerTitle(issue, job, copy, locale), locale)}</span>
          <StatusChip status={issue} />
        </p>
        {(detail || next) && <p className="routines-banner-detail">{[detail, next].filter(Boolean).join(' · ')}</p>}
        {/* The plain sentence is for reading; the backend's own words stay one click away for a fix. */}
        {raw && raw !== detail && (
          <details className="routines-banner-raw">
            <summary>{copy.banner.details}</summary>
            <p>{raw}</p>
          </details>
        )}
      </div>
      <Button disabled={busy} onClick={onRetry} variant="secondary">
        {copy.actions.retry}
      </Button>
    </div>
  )
}
