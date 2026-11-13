import { useState } from 'react'

import { ActionsMenu } from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { EmptyState } from '@/components/ui/empty-state'
import { SearchField } from '@/components/ui/search-field'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Skeleton } from '@/components/ui/skeleton'
import type { CronJob } from '@/hermes'
import { useI18n } from '@/i18n'

import { usePageFrameCopy } from '../overlays/page-frame-copy'
import { renderPanelMenuItems } from '../overlays/panel'
import { PageHeader } from '../redesign/page-header'

import {
  filterRoutines,
  ROUTINE_FILTERS,
  type RoutineFilter,
  routineStatus,
  routineTitle
} from './routine-model'
import { AgentTag, RoutineBanner, type RoutineEdit, routineMenuItems, StatusChip, useAgentLabel, useRoutineEdit } from './routine-parts'
import { capitalized, describeSchedule, formatMoment } from './routine-schedule'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'
import { routineKey, takeRoutineOverviewQuery } from './routines-store'
import type { RoutineActions } from './use-routine-actions'

interface RoutinesOverviewProps {
  source: RoutineSource
  actions: RoutineActions
  onCreate: () => void
  /** The page heading's own New routine, where the panel is a sheet (below wide). */
  headerCreate: boolean
  /** Open a routine (the rail page selects it; an agent's page shows it in place). */
  onOpen: (job: CronJob) => void
  /** One agent's list (Agent › Settings › Routines): no Agent column, no page heading. */
  agentScoped?: boolean
}

/** R1's stage: the heading, search and status filter, a banner per routine that needs attention, and
 *  every routine in one table. */
export function RoutinesOverview({ actions, agentScoped = false, headerCreate, onCreate, onOpen, source }: RoutinesOverviewProps) {
  const { t } = useI18n()
  const copy = useRoutinesCopy()
  const frame = usePageFrameCopy()
  const agentLabel = useAgentLabel()
  const [filter, setFilter] = useState<RoutineFilter>('all')
  const [query, setQuery] = useState(() => (agentScoped ? '' : takeRoutineOverviewQuery()))
  const rows = filterRoutines(source.jobs, filter, query, agentLabel)
  const failing = filterRoutines(source.jobs, 'attention', '', agentLabel)
  const agents = new Set(source.jobs.map(job => job.profile ?? '')).size
  const edit = useRoutineEdit(source)

  const newRoutine = (
    <Button onClick={onCreate} variant={agentScoped ? 'secondary' : 'default'}>
      <Codicon name="add" />
      {t.cron.newCron}
    </Button>
  )

  return (
    <div className="routines-overview">
      {!agentScoped && (
        <PageHeader actions={headerCreate ? newRoutine : undefined} description={frame.schedulesDescription} title={t.cron.title} />
      )}
      {source.loading ? (
        <OverviewSkeleton />
      ) : source.jobs.length === 0 ? (
        <EmptyState action={newRoutine} description={copy.emptyDescription} title={t.cron.emptyTitleNew} />
      ) : (
        <>
          <div className="routines-controls">
            <SearchField
              aria-label={copy.search}
              containerClassName="routines-search"
              onChange={setQuery}
              placeholder={copy.search}
              value={query}
              variant="box"
            />
            <SegmentedControl
              className="routines-filter"
              onChange={setFilter}
              options={ROUTINE_FILTERS.map(id => ({ id, label: copy.filters[id] }))}
              sizing="content"
              value={filter}
            />
            {agentScoped && <span className="routines-controls-end">{newRoutine}</span>}
          </div>
          {failing.map(job => (
            <RoutineBanner
              busy={actions.isBusy(job) || actions.isLive(job)}
                  job={job}
              key={routineKey(job)}
              onOpen={() => onOpen(job)}
              onRetry={() => void actions.runNow(job)}
              title={routineTitle(job)}
            />
          ))}
          {rows.length ? (
            <RoutinesTable actions={actions} agentScoped={agentScoped} edit={edit} jobs={rows} onOpen={onOpen} />
          ) : (
            <EmptyState description={copy.noMatchesDescription} title={copy.noMatchesTitle} />
          )}
          {!agentScoped && <p className="routines-footnote">{copy.footnote(source.jobs.length, agents)}</p>}
        </>
      )}
    </div>
  )
}

interface RoutinesTableProps {
  actions: RoutineActions
  agentScoped: boolean
  edit: RoutineEdit
  jobs: readonly CronJob[]
  onOpen: (job: CronJob) => void
}

/** Routine · Agent · When · Last run · Status · actions (Figma 1098:8480). The name opens the
 *  routine; the whole row is its click target. Narrow widths drop Agent and Last run. */
function RoutinesTable({ actions, agentScoped, edit, jobs, onOpen }: RoutinesTableProps) {
  const { locale, t } = useI18n()
  const copy = useRoutinesCopy()
  const agentLabel = useAgentLabel()
  const words = { copy, dayName: (day: string) => t.cron.days[day] ?? day, locale }

  return (
    <table className="routines-table" data-agent-scoped={agentScoped || undefined}>
      <thead>
        <tr>
          <th scope="col">{copy.columns.routine}</th>
          {!agentScoped && (
            <th data-column="agent" scope="col">
              {copy.columns.agent}
            </th>
          )}
          <th data-column="when" scope="col">
            {copy.columns.when}
          </th>
          <th data-column="last-run" scope="col">
            {copy.columns.lastRun}
          </th>
          <th scope="col">{copy.columns.status}</th>
          <th aria-label={copy.actions.more} scope="col" />
        </tr>
      </thead>
      <tbody>
        {jobs.map(job => {
          const live = actions.isLive(job)

          return (
            <tr key={routineKey(job)} onClick={() => onOpen(job)}>
              <td>
                <button className="routines-table-open" onClick={() => onOpen(job)} type="button">
                  {routineTitle(job)}
                </button>
                <span className="routines-table-sub">{describeSchedule(job.schedule, words, job.next_run_at)}</span>
              </td>
              {!agentScoped && (
                <td data-column="agent">
                  <AgentTag label={agentLabel(job.profile ?? '')} />
                </td>
              )}
              <td data-column="when">{describeSchedule(job.schedule, words, job.next_run_at)}</td>
              <td data-column="last-run">
                {live
                  ? copy.runningNow
                  : job.last_run_at
                    ? capitalized(formatMoment(job.last_run_at, locale), locale)
                    : copy.notRunYet}
              </td>
              <td>
                <StatusChip status={routineStatus(job, live)} />
              </td>
              <td onClick={event => event.stopPropagation()}>
                <ActionsMenu
                  ariaLabel={copy.actions.more}
                  contentClassName="w-44"
                  items={renderPanelMenuItems(routineMenuItems(job, actions, edit, t.cron, copy))}
                >
                  <Button aria-label={copy.actions.more} className="routines-row-menu" size="icon-compact" variant="ghost">
                    <Codicon name="ellipsis" />
                  </Button>
                </ActionsMenu>
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

/** The loaded overview's geometry while the first read is out: the controls row and table rows. */
function OverviewSkeleton() {
  return (
    <div aria-hidden className="routines-skeleton">
      <Skeleton className="routines-skeleton-controls" />
      {Array.from({ length: 5 }, (_, index) => (
        <div className="routines-skeleton-row" key={index}>
          <Skeleton className="h-3 w-40 max-w-full" />
          <Skeleton className="h-3 w-24 max-w-full" />
        </div>
      ))}
    </div>
  )
}
