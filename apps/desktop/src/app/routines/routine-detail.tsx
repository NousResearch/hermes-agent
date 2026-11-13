import { useStore } from '@nanostores/react'

import { ActionsMenu } from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { CronJob, SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { $interfaceMode } from '@/store/interface-mode'

import { jobIsScriptOnly } from '../cron/cron-job-model'
import { jobState } from '../cron/job-state'
import { catchUpChoice } from '../cron/run-control'
import { renderPanelMenuItems } from '../overlays/panel'
import { PageHeader } from '../redesign/page-header'
import { ListRow } from '../settings/primitives'
import { SettingsCard, SettingsDisclosure, SettingsOverview } from '../settings/settings-card'

import { deliveryTargets, remembersLastRun, routineStatus, routineTitle, runLimit } from './routine-model'
import { AgentTag, RoutineBanner, type RoutineEdit, routineMenuItems, StatusChip, useAgentLabel, useRoutineEdit } from './routine-parts'
import { RoutineRuns } from './routine-runs'
import { describeSchedule, formatMoment, scheduleText } from './routine-schedule'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'
import { $routinePendingDelete } from './routines-store'
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
  /** Inside an agent's Settings › Routines: the page already has its H1, so the name is a section
   *  heading there. */
  embedded?: boolean
  job: CronJob
  source: RoutineSource
  actions: RoutineActions
  onBack: () => void
  onOpenSession?: (sessionId: string, session?: SessionInfo) => void
}

/** R2: the routine on the stage. Breadcrumb back to the list, its name with Run now, Pause and "…",
 *  what it does and when and where (each with Change, into the one editor), Advanced, Runs, Delete. */
export function RoutineDetail({ actions, embedded = false, job, onBack, onOpenSession, source }: RoutineDetailProps) {
  const { t } = useI18n()
  const copy = useRoutinesCopy()
  const agentLabel = useAgentLabel()
  const title = routineTitle(job)
  const live = actions.isLive(job)
  const agent = agentLabel(job.profile ?? '')
  const status = <StatusChip status={routineStatus(job, live)} />
  const edit = useRoutineEdit(source)
  const headerActions = <RoutineHeaderActions actions={actions} edit={edit} job={job} />

  return (
    <div className="routines-detail">
      <nav aria-label={t.cron.title} className="routines-breadcrumb">
        <button onClick={onBack} type="button">
          {t.cron.title}
        </button>
        <Codicon aria-hidden name="chevron-right" size="0.75rem" />
        <span aria-current="page">{title}</span>
      </nav>
      {embedded ? (
        <div className="routines-embedded-head">
          <h2>{title}</h2>
          <span className="routines-meta">{status}</span>
          <div className="routines-embedded-actions">{headerActions}</div>
        </div>
      ) : (
        <PageHeader
          actions={headerActions}
          description={
            <span className="routines-meta">
              {status}
              <AgentTag label={agent} />
            </span>
          }
          title={title}
        />
      )}
      <SettingsOverview>
        <RoutineBanner
          busy={actions.isBusy(job) || live}
          job={job}
          onRetry={() => void actions.runNow(job)}
          title={title}
        />
        <RoutineSummary edit={edit} job={job} />
        {edit.inPlace ? (
          <RoutineAdvanced job={job} />
        ) : (
          // The rail page shows a routine and its quick actions; what it does changes in its agent's
          // settings (Figma 1098:8692).
          <SettingsCard>
            <ListRow
              action={
                <Button onClick={() => edit.open(job)} variant="secondary">
                  <Codicon name="link-external" />
                  {edit.buttonLabel(job)}
                </Button>
              }
              description={copy.editCardHint}
              title={copy.editCardTitle}
            />
          </SettingsCard>
        )}
        <RoutineRuns job={job} live={live} onOpenSession={onOpenSession} source={source} />
        {edit.inPlace && (
          <SettingsCard>
            <ListRow
              action={
                <Button onClick={() => $routinePendingDelete.set(job)} variant="destructiveOutline">
                  {copy.actions.delete}
                </Button>
              }
              description={copy.deleteHint(agent)}
              title={copy.deleteRoutine}
            />
          </SettingsCard>
        )}
      </SettingsOverview>
    </div>
  )
}

/** Run now (Stop while a run is live), Pause or Resume, and Edit / Delete behind "…". */
function RoutineHeaderActions({ actions, edit, job }: { actions: RoutineActions; edit: RoutineEdit; job: CronJob }) {
  const { t } = useI18n()
  const c = t.cron
  const copy = useRoutinesCopy()
  const busy = actions.isBusy(job)
  const paused = jobState(job) === 'paused'
  const stopping = actions.isStopping(job)

  return (
    <>
      {actions.isLive(job) ? (
        <Button disabled={stopping} onClick={() => void actions.stop(job)} variant="secondary">
          <Codicon name="debug-stop" />
          {stopping ? c.control.stopping : c.control.stop}
        </Button>
      ) : (
        <Button disabled={busy} onClick={() => void actions.runNow(job)} variant="secondary">
          <Codicon name="play" />
          {copy.actions.runNow}
        </Button>
      )}
      <Button disabled={busy} onClick={() => void actions.pauseResume(job)} variant="secondary">
        <Codicon name={paused ? 'play-circle' : 'debug-pause'} />
        {paused ? c.resumeTitle : c.pauseTitle}
      </Button>
      <ActionsMenu
        ariaLabel={copy.actions.more}
        contentClassName="w-44"
        items={renderPanelMenuItems(routineMenuItems(job, actions, edit, c, copy).slice(2))}
      >
        <Button aria-label={copy.actions.more} size="icon" variant="secondary">
          <Codicon name="ellipsis" />
        </Button>
      </ActionsMenu>
    </>
  )
}

/** What it does, when to run, where results go: in an agent's settings each row has Change into the
 *  one editor; the rail page shows them read-only (Figma 1098:8684). */
function RoutineSummary({ edit, job }: { edit: RoutineEdit; job: CronJob }) {
  const { locale, t } = useI18n()
  const copy = useRoutinesCopy()
  const deliveryLabel = useDeliveryLabel()
  const words = { copy, dayName: (day: string) => t.cron.days[day] ?? day, locale }
  const next = job.next_run_at && jobState(job) !== 'paused' ? copy.nextRun(formatMoment(job.next_run_at, locale)) : ''
  const when = [describeSchedule(job.schedule, words, job.next_run_at), next].filter(Boolean).join(' · ')

  const change = edit.inPlace ? (
    <Button onClick={() => edit.open(job)} variant="secondary">
      {copy.actions.change}
    </Button>
  ) : undefined

  return (
    <SettingsCard>
      <ListRow
        action={change}
        description={
          <span className="routines-instruction">{jobIsScriptOnly(job) && !job.prompt ? copy.scriptRoutine : job.prompt}</span>
        }
        title={copy.whatItDoes}
      />
      <ListRow action={change} description={when} title={copy.whenToRun} />
      <ListRow action={change} description={deliveryLabel(job)} title={copy.sendResultsTo} />
    </SettingsCard>
  )
}

/** Advanced (collapsed): model, missed runs, run limit, remember last run; developer mode adds the
 *  schedule expression, the script and the routine's id. */
function RoutineAdvanced({ job }: { job: CronJob }) {
  const { t } = useI18n()
  const c = t.cron
  const copy = useRoutinesCopy()
  const developer = useStore($interfaceMode) === 'advanced'
  const limit = runLimit(job)

  return (
    <SettingsDisclosure description={copy.advancedSummary} title={copy.advanced}>
      <SettingsCard>
        {!jobIsScriptOnly(job) && <ListRow description={job.model?.trim() || copy.modelDefault} title={copy.model} />}
        <ListRow description={c.control.catchUpChoices[catchUpChoice(job.catch_up)]} title={c.control.catchUpLabel} />
        <ListRow
          description={limit === null ? copy.noLimit : copy.limitProgress(job.repeat?.completed ?? 0, limit)}
          title={copy.runLimit}
        />
        <ListRow description={remembersLastRun(job) ? copy.on : copy.off} title={copy.rememberLastRun} />
        {developer && (
          <>
            <ListRow
              description={<code>{scheduleText(job.schedule, job.schedule_display ?? '')}</code>}
              title={copy.expression}
            />
            {job.script && <ListRow description={<code>{job.script}</code>} title={copy.script} />}
            <ListRow description={<code>{job.id}</code>} title={copy.routineId} />
          </>
        )}
      </SettingsCard>
    </SettingsDisclosure>
  )
}
