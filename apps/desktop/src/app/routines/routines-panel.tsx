import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { SearchField } from '@/components/ui/search-field'
import { Skeleton } from '@/components/ui/skeleton'
import type { CronJob } from '@/hermes'
import { useI18n } from '@/i18n'

import { PageSidebarCaption, PageSidebarList, PageSidebarPanel, PageSidebarRow } from '../overlays/page-sidebar'

import { filterRoutines, groupRoutines, ROUTINE_GROUPS, routineStatus, routineTitle, STATUS_TONE } from './routine-model'
import { AgentTag, useAgentLabel } from './routine-parts'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'
import { refOf, routineKey, sameRoutine, selectRoutine } from './routines-store'
import type { RoutineActions } from './use-routine-actions'

/** A short list is scanned, not searched; the field appears once it outgrows a glance (as on Agents). */
const SEARCH_FROM = 9

interface RoutinesPanelProps {
  source: RoutineSource
  actions: RoutineActions
  /** The routine on the stage. */
  shown: CronJob | null
  onCreate: () => void
}

/** The page's panel (rev 1383:21029): New routine, then every routine under NEEDS ATTENTION /
 *  SCHEDULED / PAUSED (/ FINISHED), each with its status dot and agent, as Chats lists chats. Below
 *  wide it is the band menu's sheet. */
export function RoutinesPanel({ actions, onCreate, shown, source }: RoutinesPanelProps) {
  const { t } = useI18n()

  return (
    <PageSidebarPanel
      action={
        <Button onClick={onCreate} variant="secondary">
          <Codicon name="add" />
          {t.cron.newCron}
        </Button>
      }
      title={t.cron.title}
    >
      <RoutineGroups actions={actions} shown={shown} source={source} />
    </PageSidebarPanel>
  )
}

interface RoutineGroupsProps {
  source: RoutineSource
  actions: RoutineActions
  shown: CronJob | null
}

/** The routines by state, with a search once the list is long: the panel's body, and the stage's
 *  own list below wide while no routine is open. */
export function RoutineGroups({ actions, shown, source }: RoutineGroupsProps) {
  const copy = useRoutinesCopy()
  const agentLabel = useAgentLabel()
  const [query, setQuery] = useState('')
  const groups = groupRoutines(filterRoutines(source.jobs, query, agentLabel))
  const searching = source.jobs.length >= SEARCH_FROM

  if (source.loading) {
    return (
      <div aria-hidden className="routines-panel-skeleton">
        {Array.from({ length: 4 }, (_, index) => (
          <Skeleton className="h-3.5 w-40 max-w-full" key={index} />
        ))}
      </div>
    )
  }

  return (
    <>
      {searching && (
        <SearchField
          aria-label={copy.search}
          containerClassName="page-sidebar-search"
          onChange={setQuery}
          placeholder={copy.search}
          value={query}
          variant="box"
        />
      )}
      {searching && query && ROUTINE_GROUPS.every(group => !groups[group].length) && (
        <p className="routines-panel-empty">{copy.noMatchesDescription}</p>
      )}
      {ROUTINE_GROUPS.map(group =>
        groups[group].length ? (
          <PageSidebarList key={group}>
            <PageSidebarCaption>{`${copy.groups[group]} · ${groups[group].length}`}</PageSidebarCaption>
            {groups[group].map(job => {
              const status = routineStatus(job, actions.isLive(job))

              return (
                <PageSidebarRow
                  current={sameRoutine(job, shown && refOf(shown))}
                  key={routineKey(job)}
                  onClick={() => selectRoutine(refOf(job))}
                >
                  <span aria-hidden className="page-sidebar-dot routines-dot" data-tone={STATUS_TONE[status]} />
                  <span className="min-w-0 flex-1 truncate">{routineTitle(job)}</span>
                  <span className="sr-only">{copy.status[status]}</span>
                  <AgentTag label={agentLabel(job.profile ?? '')} />
                </PageSidebarRow>
              )
            })}
          </PageSidebarList>
        ) : null
      )}
    </>
  )
}
