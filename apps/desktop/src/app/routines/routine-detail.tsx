import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import type { CronJob, SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'

import { jobIsScriptOnly } from '../cron/cron-job-model'
import { PageHeader } from '../redesign/page-header'

import { deliveryTargets, routinePaused, routineSpent, routineStatus, routineTitle } from './routine-model'
import { RoutineBanner, StatusChip, useAgentLabel } from './routine-parts'
import { RoutineRuns } from './routine-runs'
import { capitalized, describeSchedule, formatMoment } from './routine-schedule'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'
import { openRoutineSheet, refOf } from './routines-store'
import type { RoutineActions } from './use-routine-actions'

/** Where results go, in words: each channel's name, the agent's own chat, or "kept in Runs only". */
export function useDeliveryLabel(): (job: CronJob) => string {
  const { t } = useI18n()
  const copy = useRoutinesCopy()
  const agentLabel = useAgentLabel()

  return job => {
    const targets = deliveryTargets(job.deliver)

    if (!targets.length) {
      return copy.keptInRuns
    }

    return targets
      .map(target => {
        if (target === 'bot-chat' || target.startsWith('bot-chat:')) {
          return copy.agentChat(agentLabel(target.slice('bot-chat:'.length) || job.profile || ''))
        }

        return t.cron.deliveryLabels[target] ?? target
      })
      .join(' · ')
  }
}

interface RoutineDetailProps {
  job: CronJob
  source: RoutineSource
  actions: RoutineActions
  /** Below wide the breadcrumb's first crumb goes back to the list. */
  onBack?: () => void
  onOpenSession?: (sessionId: string, session?: SessionInfo) => void
}

/** One routine on the stage (rev 1383:21029): breadcrumb, its name and status with Run now, Pause or
 *  Resume and Edit; the failed-run banner; one card with its instruction beside when it runs and where
 *  results go; and its run history. */
export function RoutineDetail({ actions, job, onBack, onOpenSession, source }: RoutineDetailProps) {
  const { t } = useI18n()
  const agentLabel = useAgentLabel()
  const title = routineTitle(job)
  const live = actions.isLive(job)

  return (
    <div className="routines-detail">
      <nav aria-label={t.cron.title} className="routines-breadcrumb">
        {onBack ? (
          <button onClick={onBack} type="button">
            {t.cron.title}
          </button>
        ) : (
          <span>{t.cron.title}</span>
        )}
        <Codicon aria-hidden name="chevron-right" size="0.75rem" />
        <span aria-current="page">{agentLabel(job.profile ?? '')}</span>
      </nav>
      <PageHeader
        actions={<RoutineHeaderActions actions={actions} job={job} />}
        className="routines-detail-header"
        title={
          <>
            <span className="routines-title">{title}</span>
            <StatusChip status={routineStatus(job, live)} />
          </>
        }
      />
      <div className="routines-detail-body">
        <RoutineBanner busy={actions.isBusy(job) || live} job={job} onRetry={() => void actions.runNow(job)} />
        <RoutineSummary job={job} />
        <RoutineRuns job={job} live={live} onOpenSession={onOpenSession} source={source} />
      </div>
    </div>
  )
}

/** Run now (Run routine while paused; Stop while a run is live), Pause or Resume, and Edit, each with
 *  its tooltip on hover and on keyboard focus (rev 1406:23973). */
function RoutineHeaderActions({ actions, job }: { actions: RoutineActions; job: CronJob }) {
  const { t } = useI18n()
  const c = t.cron
  const copy = useRoutinesCopy()
  const busy = actions.isBusy(job)
  const paused = routinePaused(job)
  const stopping = actions.isStopping(job)
  const toggle = paused ? c.resume : c.pause

  return (
    <>
      {actions.isLive(job) ? (
        <Button disabled={stopping} onClick={() => void actions.stop(job)} variant="secondary">
          <Codicon name="debug-stop" />
          {stopping ? c.control.stopping : c.control.stop}
        </Button>
      ) : (
        <Tip label={copy.actions.runNowHint}>
          <Button disabled={busy} onClick={() => void actions.runNow(job)}>
            {paused ? copy.actions.runRoutine : copy.actions.runNow}
          </Button>
        </Tip>
      )}
      {/* A routine with no runs left has nothing to pause or resume: Run now starts it again. */}
      {!routineSpent(job) && (
        <Tip label={toggle}>
          <Button
            aria-label={toggle}
            disabled={busy}
            onClick={() => void actions.pauseResume(job)}
            size="icon"
            variant="secondary"
          >
            <Codicon name={paused ? 'play' : 'debug-pause'} />
          </Button>
        </Tip>
      )}
      <Tip label={c.edit}>
        <Button
          aria-label={c.edit}
          onClick={() => openRoutineSheet({ mode: 'edit', ref: refOf(job) })}
          size="icon"
          variant="secondary"
        >
          <Codicon name="edit" />
        </Button>
      </Tip>
    </>
  )
}

/** Past this many lines an instruction folds, with a way to read the rest. */
const INSTRUCTION_LINES = 8

/** The summary card: what it does on the left; how often, its last and next run and where results go
 *  on the right (one column on compact). Times read in the app's own words ("Yesterday, 7:00 AM"). */
function RoutineSummary({ job }: { job: CronJob }) {
  const { locale, t } = useI18n()
  const copy = useRoutinesCopy()
  const deliveryLabel = useDeliveryLabel()
  const [expanded, setExpanded] = useState(false)
  const words = { copy, dayName: (day: string) => t.cron.days[day] ?? day, locale }
  const instruction = jobIsScriptOnly(job) && !job.prompt ? copy.scriptRoutine : (job.prompt ?? '')
  const long = instruction.split('\n').length > INSTRUCTION_LINES || instruction.length > INSTRUCTION_LINES * 80

  const next = routinePaused(job)
    ? copy.status.paused
    : job.next_run_at
      ? capitalized(formatMoment(job.next_run_at, locale), locale)
      : copy.notScheduled

  const facts: [string, string][] = [
    [copy.frequency, describeSchedule(job.schedule, words, job.next_run_at)],
    [copy.lastRan, job.last_run_at ? capitalized(formatMoment(job.last_run_at, locale), locale) : copy.notRunYet],
    [copy.nextRunLabel, next],
    [copy.sendResultsTo, deliveryLabel(job)]
  ]

  return (
    <section aria-label={routineTitle(job)} className="routines-summary">
      <div className="routines-summary-instruction">
        <h2>{copy.instruction}</h2>
        <p data-folded={long && !expanded ? '' : undefined}>{instruction}</p>
        {long && (
          <Button onClick={() => setExpanded(!expanded)} size="inline" variant="textStrong">
            {expanded ? t.common.collapse : t.common.expand}
          </Button>
        )}
      </div>
      <dl className="routines-summary-facts">
        {facts.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
