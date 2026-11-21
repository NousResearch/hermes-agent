import './routines.css'

import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { CronJob } from '@/hermes'
import { useI18n } from '@/i18n'

import type { AgentSettingsOwner } from '../agent-settings/data'
import { agentOverviewQueries } from '../agent-settings/overview-queries'
import { CRON_ROUTE } from '../routes'
import { SettingsCard, SettingsCardSkeleton, SettingsEmptyRow, SettingsLinkRow } from '../settings/settings-card'

import { compareRoutines, routineStatus, routineTitle, STATUS_TONE } from './routine-model'
import { describeSchedule } from './routine-schedule'
import { useRoutinesCopy } from './routines-copy'
import { openRoutineSheet, refOf, selectRoutine } from './routines-store'

/** The Routines page is the only place a routine is created or changed: an agent's surfaces link
 *  there, with that routine open or a new one started for that agent. An agent page only shows on
 *  the machine this window is on (another machine's agent waits behind "Open agent"), which is the
 *  machine the Routines page lists. */
export function useRoutinesPageLinks(owner: AgentSettingsOwner) {
  const navigate = useNavigate()

  return {
    create: () => {
      openRoutineSheet({ mode: 'create', profile: owner.profile })
      navigate(CRON_ROUTE)
    },
    open: (job: CronJob) => {
      selectRoutine(refOf({ ...job, profile: job.profile ?? owner.profile }))
      navigate(CRON_ROUTE)
    }
  }
}

/** One agent's routines, read from its own machine. */
export function useAgentRoutineList(owner: AgentSettingsOwner) {
  return useQuery(agentOverviewQueries.routines(owner))
}

/** Agent › Settings › Routines in the redesign: the agent's routines, read only, each opening on the
 *  Routines page, and New routine starting one there for this agent. */
export function AgentRoutines({ owner }: { owner: AgentSettingsOwner }) {
  const { locale, t } = useI18n()
  const copy = useRoutinesCopy()
  const links = useRoutinesPageLinks(owner)
  const routines = useAgentRoutineList(owner)
  const words = { copy, dayName: (day: string) => t.cron.days[day] ?? day, locale }
  const jobs = [...(routines.data ?? [])].sort(compareRoutines)

  return (
    <div className="routines-agent-list">
      <div className="routines-agent-list-head">
        <p className="routines-field-note">{copy.agentRoutinesHint}</p>
        <Button onClick={links.create} variant="secondary">
          <Codicon name="add" />
          {t.cron.newCron}
        </Button>
      </div>
      {routines.isPending ? (
        <SettingsCardSkeleton rows={2} />
      ) : (
        <SettingsCard>
          {routines.isError && <SettingsEmptyRow title={t.cron.failedLoad} />}
          {!routines.isError && jobs.length === 0 && <SettingsEmptyRow title={t.cron.emptyTitleNew} />}
          {jobs.map(job => (
            <SettingsLinkRow
              description={describeSchedule(job.schedule, words, job.next_run_at)}
              key={job.id}
              meta={
                <span className="routines-chip" data-tone={STATUS_TONE[routineStatus(job, false)]}>
                  {copy.status[routineStatus(job, false)]}
                </span>
              }
              onClick={() => links.open(job)}
              title={routineTitle(job)}
            />
          ))}
        </SettingsCard>
      )}
    </div>
  )
}
