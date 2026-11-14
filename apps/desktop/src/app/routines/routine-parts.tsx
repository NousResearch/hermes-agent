import { useStore } from '@nanostores/react'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { type CronJob, getApiRequestConnection } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { $profiles, profileLabel } from '@/store/profile'

import { jobState } from '../cron/job-state'
import type { PanelMenuItem } from '../overlays/panel'
import { SETTINGS_ROUTE } from '../routes'

import { plainRunError, routineErrorText } from './routine-error'
import { type RoutineIssue, routineIssue, type RoutineStatus, STATUS_TONE } from './routine-model'
import { formatMoment } from './routine-schedule'
import type { RoutineSource } from './routine-source'
import { type RoutinesCopy, useRoutinesCopy } from './routines-copy'
import { $routinePendingDelete, $routineRename, openRoutineEditor } from './routines-store'
import type { RoutineActions } from './use-routine-actions'

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

const BANNER_TITLE: Record<RoutineIssue, (copy: RoutinesCopy, name: string, job: CronJob) => string> = {
  delivery: (copy, name) => copy.banner.delivery(name),
  failed: (copy, name, job) =>
    (job.failure_streak ?? 0) > 1 ? copy.banner.failedStreak(name, job.failure_streak ?? 0) : copy.banner.failedOnce(name),
  overdue: (copy, name) => copy.banner.overdue(name)
}

interface RoutineBannerProps {
  job: CronJob
  title: string
  busy: boolean
  onRetry: () => void
  onOpen?: () => void
}

/** A routine that needs attention (Figma 1098:8471): what went wrong in one plain sentence, when it
 *  will try again, and Retry. Shown only while there is a problem. */
export function RoutineBanner({ busy, job, onOpen, onRetry, title }: RoutineBannerProps) {
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
      : job.next_run_at && jobState(job) !== 'paused'
        ? copy.banner.retryAt(formatMoment(job.next_run_at, locale))
        : ''

  const heading = BANNER_TITLE[issue](copy, title, job)
  const raw = routineErrorText(job)
  const detail = raw ? plainRunError(raw, copy, agentLabel(job.profile ?? '')) : ''

  return (
    <div className="routines-banner" role="status">
      <Codicon aria-hidden className="routines-banner-icon" name="warning" />
      <div className="routines-banner-text">
        {onOpen ? (
          <button className="routines-banner-title" onClick={onOpen} type="button">
            {heading}
          </button>
        ) : (
          <p className="routines-banner-title">{heading}</p>
        )}
        {(detail || next) && <p className="routines-banner-detail">{[detail, next].filter(Boolean).join(' · ')}</p>}
        {/* The plain sentence is for reading; the backend's own words stay one click away for a fix. */}
        {raw && raw !== detail && (
          <details className="routines-banner-raw">
            <summary>{copy.banner.details}</summary>
            <p>{raw}</p>
          </details>
        )}
      </div>
      <Button disabled={busy} onClick={onRetry}>
        {copy.actions.retry}
      </Button>
    </div>
  )
}

/** A routine inside its agent's Settings › Routines; `edit` opens it in the editor there. */
export function routineSettingsHref(job: CronJob, connectionId: null | string, edit: boolean): string {
  const params = new URLSearchParams({
    agentProfile: job.profile || 'default',
    panel: 'settings',
    routine: job.id,
    tab: 'agent:schedules'
  })

  if (connectionId) {
    params.set('connection', connectionId)
  }

  if (edit) {
    params.set('edit', '1')
  }

  return `${SETTINGS_ROUTE}?${params}`
}

/** How a surface edits: in place (an agent's Settings › Routines), or by going there (the rail). */
export interface RoutineEdit {
  inPlace: boolean
  /** "Edit in <agent> settings" on the rail, "Edit" in place. */
  menuLabel: (job: CronJob) => string
  /** The button that takes the rail to the editor ("Edit in <agent>"). */
  buttonLabel: (job: CronJob) => string
  open: (job: CronJob) => void
}

export function useRoutineEdit(source: RoutineSource): RoutineEdit {
  const navigate = useNavigate()
  const copy = useRoutinesCopy()
  const agentLabel = useAgentLabel()
  const agent = (job: CronJob) => agentLabel(job.profile ?? '')

  if (source.editsInPlace) {
    return {
      buttonLabel: () => copy.actions.edit,
      inPlace: true,
      menuLabel: () => copy.actions.edit,
      open: job => openRoutineEditor({ job, mode: 'edit' })
    }
  }

  return {
    buttonLabel: job => copy.editInAgent(agent(job)),
    inPlace: false,
    menuLabel: job => copy.editInSettings(agent(job)),
    open: job => navigate(routineSettingsHref(job, source.connectionId ?? getApiRequestConnection(), true))
  }
}

/** The routine actions every row, the detail's "…" and right-click share. The rail page's are its quick
 *  actions (Run now, Pause, Rename, Edit in the agent's settings, Delete; Figma 1098:8615). */
export function routineMenuItems(
  job: CronJob,
  actions: RoutineActions,
  edit: RoutineEdit,
  c: Translations['cron'],
  copy: RoutinesCopy
): PanelMenuItem[] {
  const paused = jobState(job) === 'paused'
  const busy = actions.isBusy(job)

  return [
    { disabled: busy || actions.isLive(job), icon: 'play', label: copy.actions.runNow, onSelect: () => void actions.runNow(job) },
    {
      disabled: busy,
      icon: paused ? 'play-circle' : 'debug-pause',
      label: paused ? c.resumeTitle : c.pauseTitle,
      onSelect: () => void actions.pauseResume(job)
    },
    ...(edit.inPlace ? [] : [{ icon: 'rename', label: copy.rename, onSelect: () => $routineRename.set(job) }]),
    { icon: edit.inPlace ? 'edit' : 'link-external', label: edit.menuLabel(job), onSelect: () => edit.open(job) },
    { icon: 'trash', label: copy.actions.delete, onSelect: () => $routinePendingDelete.set(job), tone: 'danger' }
  ]
}
