import './routines.css'

import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { EmptyState } from '@/components/ui/empty-state'
import type { CronJob, SessionInfo } from '@/hermes'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useI18n } from '@/i18n'

import { useJobFocus } from '../cron/use-job-focus'
import { PAGE_SIDEBAR_NARROW } from '../overlays/page-sidebar'
import { Panel } from '../overlays/panel'
import { PageHeader } from '../redesign/page-header'

import { RoutineDetail } from './routine-detail'
import { $heldRoutine } from './routine-held-delete'
import { neighbourInList, routinesInPanelOrder } from './routine-model'
import { RoutineSheet } from './routine-sheet'
import { type RoutineSource, useRailRoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'
import { RoutineGroups, RoutinesPanel } from './routines-panel'
import {
  $routineSelection,
  closeRoutineSheet,
  openRoutineSheet,
  refOf,
  routineKey,
  sameRoutine,
  selectRoutine
} from './routines-store'
import { type HeldDelete, useHeldDelete } from './use-held-delete'
import { type RoutineActions, useRoutineActions } from './use-routine-actions'

interface RoutinesPageProps {
  onClose: () => void
  onOpenSession?: (sessionId: string, session?: SessionInfo) => void
}

const openJob = (job: CronJob) => selectRoutine(refOf(job))

/** The routine the stage shows: the one selected, else (wide) the first in the list. Below wide, with
 *  none selected, the stage is the list itself, as the panel is a sheet there. */
function useShownRoutine(source: RoutineSource, narrow: boolean): CronJob | null {
  const selection = useStore($routineSelection)
  const selected = selection ? source.jobs.find(job => sameRoutine(job, selection)) : undefined

  if (selected) {
    return selected
  }

  return narrow ? null : (routinesInPanelOrder(source.jobs)[0] ?? null)
}

/** The rail's Routines destination in the redesign (rev 1081:29252): every agent's routines in the
 *  panel by state, one routine on the stage, and the one sheet that creates and edits them. Classic
 *  keeps CronView. */
export function RoutinesPage({ onClose, onOpenSession }: RoutinesPageProps) {
  const { t } = useI18n()
  const source = useRailRoutineSource()
  const actions = useRoutineActions(source)
  const held = useHeldDelete(source)
  const narrow = useMediaQuery(PAGE_SIDEBAR_NARROW)
  const shown = useShownRoutine(source, narrow)

  // A one-shot "open this routine" (a notification, a sidebar link), by id or name.
  useJobFocus(source.jobs, !source.loading, openJob)

  // The sheet is store state, not page state: leaving Routines with one open must not leave an empty
  // create/edit sheet waiting to pop back up on the next visit.
  useEffect(() => () => closeRoutineSheet(), [])

  const create = () => openRoutineSheet({ mode: 'create', profile: null })

  // Delete lands on the next routine in the list (the one before it at the end), then holds.
  const remove = (job: CronJob) => {
    const next = neighbourInList(routinesInPanelOrder(source.jobs), candidate => sameRoutine(candidate, refOf(job)))
    selectRoutine(next ? refOf(next) : null)
    void held.hold(job)
  }

  return (
    <Panel closeLabel={t.cron.close} contentClassName="routines-page" onClose={onClose}>
      <RoutinesPanel actions={actions} onCreate={create} shown={shown} source={source} />
      {/* Keyed by what it shows, so opening a routine starts at its top, not the last one's scroll. */}
      <div className="routines-stage page-stage" data-scrolls="" key={shown ? routineKey(shown) : 'list'}>
        <DeletedBanner held={held} />
        <RoutinesStage
          actions={actions}
          narrow={narrow}
          onCreate={create}
          onOpenSession={onOpenSession}
          shown={shown}
          source={source}
        />
      </div>
      <RoutineSheet onDelete={remove} source={source} />
    </Panel>
  )
}

interface RoutinesStageProps {
  source: RoutineSource
  actions: RoutineActions
  shown: CronJob | null
  narrow: boolean
  onCreate: () => void
  onOpenSession?: (sessionId: string, session?: SessionInfo) => void
}

function RoutinesStage({ actions, narrow, onCreate, onOpenSession, shown, source }: RoutinesStageProps) {
  const { t } = useI18n()
  const copy = useRoutinesCopy()

  const newRoutine = (
    <Button onClick={onCreate}>
      <Codicon name="add" />
      {t.cron.newCron}
    </Button>
  )

  if (shown) {
    return (
      <RoutineDetail
        actions={actions}
        job={shown}
        onBack={narrow ? () => selectRoutine(null) : undefined}
        onOpenSession={onOpenSession}
        source={source}
      />
    )
  }

  if (!source.loading && source.jobs.length === 0) {
    return (
      <div className="routines-detail">
        <PageHeader title={t.cron.title} />
        <EmptyState action={newRoutine} description={copy.emptyDescription} title={t.cron.emptyTitleNew} />
      </div>
    )
  }

  // Below wide the panel is a sheet: the stage lists the routines until one is opened.
  return narrow ? (
    <div className="routines-detail">
      <PageHeader actions={newRoutine} title={t.cron.title} />
      <RoutineGroups actions={actions} shown={null} source={source} />
    </div>
  ) : (
    <div aria-busy className="routines-detail" />
  )
}

/** "<name> deleted." with Undo and ×, at the stage's top above the breadcrumb: a change to the list,
 *  not to the routine on screen (rev 1406:24625). It stays until it is answered. */
function DeletedBanner({ held }: { held: HeldDelete }) {
  const { t } = useI18n()
  const copy = useRoutinesCopy()
  const routine = useStore($heldRoutine)

  if (!routine) {
    return null
  }

  return (
    <div className="routines-deleted" role="status">
      <Codicon aria-hidden className="routines-deleted-icon" name="pass" />
      <p>{copy.deleted(routine.name)}</p>
      <Button onClick={() => void held.undo()} size="sm" variant="secondary">
        {copy.actions.undo}
      </Button>
      <Button aria-label={t.common.close} onClick={() => void held.dismiss()} size="icon-sm" variant="ghost">
        <Codicon name="close" />
      </Button>
    </div>
  )
}
