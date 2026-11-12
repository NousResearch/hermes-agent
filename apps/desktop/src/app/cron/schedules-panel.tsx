import './schedules.css'

import type { ReactNode } from 'react'

import { PageLoader } from '@/components/page-loader'
import { ActionsContextMenu } from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import type { AutomationBlueprint, CronJob } from '@/types/cron'

import { PageSidebarCaption, PageSidebarList, PageSidebarPanel, PageSidebarRow } from '../overlays/page-sidebar'
import { type PanelMenuItem, PanelRowMenu, renderPanelMenuItems } from '../overlays/panel'

import { CronEmptyPage, CronListEmpty, CronPageHeader } from './empty-page'
import { jobState, jobTitle, STATE_DOT } from './job-state'

/** A short list is scanned, not searched; the field appears once it outgrows a glance (as on Agents). */
const SEARCH_FROM = 9

interface CronSchedulesPanelProps {
  blueprints: readonly AutomationBlueprint[]
  /** The schedules the search leaves, in list order. */
  jobs: readonly CronJob[]
  menuItems: (job: CronJob) => PanelMenuItem[]
  onCreate: () => void
  onQueryChange: (query: string) => void
  onSelect: (jobId: string) => void
  onStartFrom: (blueprintKey: string) => void
  query: string
  selectedId: null | string
  totalCount: number
}

/** The Schedules destination's list in the page sidebar panel: New schedule, the schedules under
 *  their count, then the blueprints to start another from. The stage shows the selected schedule. */
export function CronSchedulesPanel({
  blueprints,
  jobs,
  menuItems,
  onCreate,
  onQueryChange,
  onSelect,
  onStartFrom,
  query,
  selectedId,
  totalCount
}: CronSchedulesPanelProps) {
  const { t } = useI18n()
  const c = t.cron

  const newSchedule = (
    <Button onClick={onCreate} size="sm" variant="secondary">
      <Codicon name="add" />
      {c.newCron}
    </Button>
  )

  return (
    <PageSidebarPanel action={newSchedule} title={c.title}>
      {totalCount >= SEARCH_FROM && (
        <SearchField
          aria-label={c.search}
          containerClassName="page-sidebar-search"
          onChange={onQueryChange}
          placeholder={c.search}
          value={query}
          variant="box"
        />
      )}
      {totalCount > 0 && (
        <PageSidebarList>
          <PageSidebarCaption>{c.count(jobs.length)}</PageSidebarCaption>
          {jobs.map(job => (
            <ScheduleRow
              current={job.id === selectedId}
              job={job}
              key={job.id}
              menuItems={menuItems(job)}
              menuLabel={c.manage}
              onSelect={() => onSelect(job.id)}
            />
          ))}
        </PageSidebarList>
      )}
      {/* With no schedule yet the stage lists the blueprints as starting points; one list is enough. */}
      {totalCount > 0 && blueprints.length > 0 && (
        <PageSidebarList>
          <PageSidebarCaption>{c.blueprints.tab}</PageSidebarCaption>
          {blueprints.map(item => (
            <PageSidebarRow key={item.key} onClick={() => onStartFrom(item.key)}>
              <Codicon className="page-sidebar-glyph" name="rocket" size="1rem" />
              <span className="min-w-0 flex-1 truncate">{item.title}</span>
              {item.plugin && <span className="page-sidebar-meta">{item.plugin}</span>}
            </PageSidebarRow>
          ))}
        </PageSidebarList>
      )}
    </PageSidebarPanel>
  )
}

interface ScheduleRowProps {
  current: boolean
  job: CronJob
  menuItems: PanelMenuItem[]
  menuLabel: string
  onSelect: () => void
}

/** A schedule: its state dot and name; Edit and Delete on the end glyph and on right-click. */
function ScheduleRow({ current, job, menuItems, menuLabel, onSelect }: ScheduleRowProps) {
  return (
    <ActionsContextMenu ariaLabel={menuLabel} contentClassName="w-40" items={renderPanelMenuItems(menuItems)}>
      <div className="page-sidebar-item group/row" data-panel-row={job.id}>
        <PageSidebarRow current={current} onClick={onSelect}>
          <span
            aria-hidden="true"
            className={cn('page-sidebar-dot', STATE_DOT[jobState(job)] ?? 'bg-muted-foreground')}
          />
          <span className="min-w-0 flex-1 truncate">{jobTitle(job)}</span>
        </PageSidebarRow>
        <div className="page-sidebar-item-menu">
          <PanelRowMenu items={menuItems} label={menuLabel} />
        </div>
      </div>
    </ActionsContextMenu>
  )
}

interface CronStageProps {
  blueprints: readonly AutomationBlueprint[]
  /** The selected schedule's heading and detail; absent when nothing is listed. */
  children?: ReactNode
  loading: boolean
  onCreate: () => void
  onStartFrom: (blueprintKey: string) => void
  totalCount: number
}

/** The stage beside the panel: the selected schedule, else the page heading over the empty card
 *  (blueprints offered as starting points) or, when a search leaves nothing, the empty search. */
export function CronStage({ blueprints, children, loading, onCreate, onStartFrom, totalCount }: CronStageProps) {
  const { t } = useI18n()

  if (loading) {
    return (
      <div className="cron-stage">
        <PageLoader label={t.cron.loading} />
      </div>
    )
  }

  return (
    <div className="cron-stage">
      {children || (
        <>
          <CronPageHeader totalCount={totalCount} />
          {totalCount === 0 ? (
            <CronEmptyPage blueprints={blueprints} onCreate={onCreate} onStartFrom={onStartFrom} />
          ) : (
            <CronListEmpty searching />
          )}
        </>
      )}
    </div>
  )
}
