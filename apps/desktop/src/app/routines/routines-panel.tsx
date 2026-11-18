import { useState } from 'react'

import { ActionsContextMenu } from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { SearchField } from '@/components/ui/search-field'
import { Skeleton } from '@/components/ui/skeleton'
import type { CronJob } from '@/hermes'
import { useI18n } from '@/i18n'

import { PageSidebarCaption, PageSidebarList, PageSidebarPanel, PageSidebarRow } from '../overlays/page-sidebar'
import { PanelRowMenu, renderPanelMenuItems } from '../overlays/panel'

import { filterRoutines, groupRoutines, ROUTINE_GROUPS, routineStatus, routineTitle, STATUS_TONE } from './routine-model'
import { AgentTag, type RoutineEdit, routineMenuItems, useAgentLabel, useRoutineEdit } from './routine-parts'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'
import { routineKey, type RoutineRef, sameRoutine, selectRoutine } from './routines-store'
import type { RoutineActions } from './use-routine-actions'

/** A short list is scanned, not searched; the field appears once it outgrows a glance (as on Agents). */
const SEARCH_FROM = 9

interface RoutinesPanelProps {
  source: RoutineSource
  actions: RoutineActions
  selection: null | RoutineRef
  onCreate: () => void
}

/** R1's panel: New routine, the overview row, then the routines under NEEDS ATTENTION / SCHEDULED /
 *  PAUSED (/ FINISHED), each with its status dot and agent. Below wide it is the band menu's sheet. */
export function RoutinesPanel({ actions, onCreate, selection, source }: RoutinesPanelProps) {
  const { t } = useI18n()
  const copy = useRoutinesCopy()
  const agentLabel = useAgentLabel()
  const [query, setQuery] = useState('')
  const shown = filterRoutines(source.jobs, 'all', query, agentLabel)
  const groups = groupRoutines(shown)
  const edit = useRoutineEdit(source)

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
      {source.jobs.length >= SEARCH_FROM && (
        <SearchField
          aria-label={copy.search}
          containerClassName="page-sidebar-search"
          onChange={setQuery}
          placeholder={copy.search}
          value={query}
          variant="box"
        />
      )}
      <PageSidebarList>
        <PageSidebarRow current={!selection} onClick={() => selectRoutine(null)}>
          <Codicon className="page-sidebar-glyph" name="layout" size="1rem" />
          <span className="min-w-0 flex-1 truncate">{copy.allRoutines}</span>
        </PageSidebarRow>
      </PageSidebarList>
      {source.loading ? (
        <div aria-hidden className="routines-panel-skeleton">
          {Array.from({ length: 4 }, (_, index) => (
            <Skeleton className="h-3.5 w-40 max-w-full" key={index} />
          ))}
        </div>
      ) : (
        ROUTINE_GROUPS.map(group =>
          groups[group].length ? (
            <PageSidebarList key={group}>
              <PageSidebarCaption>{`${copy.groups[group]} · ${groups[group].length}`}</PageSidebarCaption>
              {groups[group].map(job => (
                <RoutineRow
                  actions={actions}
                  agent={agentLabel(job.profile ?? '')}
                  current={sameRoutine(job, selection)}
                  edit={edit}
                  job={job}
                  key={routineKey(job)}
                />
              ))}
            </PageSidebarList>
          ) : null
        )
      )}
    </PageSidebarPanel>
  )
}

interface RoutineRowProps {
  actions: RoutineActions
  agent: string
  current: boolean
  edit: RoutineEdit
  job: CronJob
}

function RoutineRow({ actions, agent, current, edit, job }: RoutineRowProps) {
  const { t } = useI18n()
  const copy = useRoutinesCopy()
  const items = routineMenuItems(job, actions, edit, t.cron, copy)
  const tone = STATUS_TONE[routineStatus(job, actions.isLive(job))]

  return (
    <ActionsContextMenu ariaLabel={copy.actions.more} contentClassName="w-44" items={renderPanelMenuItems(items)}>
      <div className="page-sidebar-item routines-panel-item group/row">
        <PageSidebarRow
          current={current}
          onClick={() => selectRoutine({ id: job.id, profile: job.profile ?? '' })}
        >
          <span aria-hidden className="page-sidebar-dot routines-dot" data-tone={tone} />
          <span className="min-w-0 flex-1 truncate">{routineTitle(job)}</span>
          <AgentTag label={agent} />
        </PageSidebarRow>
        <div className="page-sidebar-item-menu">
          <PanelRowMenu items={items} label={copy.actions.more} />
        </div>
      </div>
    </ActionsContextMenu>
  )
}
