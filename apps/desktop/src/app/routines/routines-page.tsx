import './routines.css'

import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import type { CronJob, SessionInfo } from '@/hermes'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useI18n } from '@/i18n'
import { $cronFocusJobId, setCronFocusJobId } from '@/store/cron'
import { $activeGatewayProfile } from '@/store/profile'

import { PAGE_SIDEBAR_NARROW } from '../overlays/page-sidebar'
import { Panel } from '../overlays/panel'

import { RoutineDetail } from './routine-detail'
import { RoutineDialogs } from './routine-dialogs'
import { useRailRoutineSource } from './routine-source'
import { RoutinesOverview } from './routines-overview'
import { RoutinesPanel } from './routines-panel'
import { $routineSelection, openRoutineEditor, routineKey, sameRoutine, selectRoutine } from './routines-store'
import { useRoutineActions } from './use-routine-actions'

interface RoutinesPageProps {
  onClose: () => void
  onOpenSession?: (sessionId: string, session?: SessionInfo) => void
}

const refOf = (job: CronJob) => ({ id: job.id, profile: job.profile ?? '' })

/** A one-shot "open this routine" (a notification, an old sidebar link) by id or name. */
function useFocusRequest(jobs: readonly CronJob[]) {
  const focus = useStore($cronFocusJobId)

  useEffect(() => {
    if (!focus) {
      return
    }

    const match = jobs.find(job => job.id === focus || job.name === focus)

    if (match) {
      selectRoutine(refOf(match))
      setCronFocusJobId(null)
    }
  }, [focus, jobs])
}

/** The rail's Routines destination in the redesign (R1/R2): every agent's routines in the page panel,
 *  the overview or one routine on the stage, and the one editor. Classic keeps CronView. */
export function RoutinesPage({ onClose, onOpenSession }: RoutinesPageProps) {
  const { t } = useI18n()
  const source = useRailRoutineSource()
  const actions = useRoutineActions(source)
  const selection = useStore($routineSelection)
  const activeProfile = useStore($activeGatewayProfile)
  const narrow = useMediaQuery(PAGE_SIDEBAR_NARROW)
  const selected = selection ? (source.jobs.find(job => sameRoutine(job, selection)) ?? null) : null

  useFocusRequest(source.jobs)

  const create = () => openRoutineEditor({ lockProfile: false, mode: 'pick', profile: activeProfile || 'default' })

  return (
    <Panel closeLabel={t.cron.close} contentClassName="routines-page" onClose={onClose}>
      <RoutinesPanel actions={actions} onCreate={create} selection={selected ? selection : null} source={source} />
      {/* Keyed by what it shows, so opening a routine starts at its top, not the list's scroll. */}
      <div className="routines-stage" key={selected ? routineKey(selected) : 'overview'}>
        {selected ? (
          <RoutineDetail
            actions={actions}
            job={selected}
            onBack={() => selectRoutine(null)}
            onOpenSession={onOpenSession}
            source={source}
          />
        ) : (
          <RoutinesOverview
            actions={actions}
            headerCreate={narrow}
            onCreate={create}
            onOpen={job => selectRoutine(refOf(job))}
            source={source}
          />
        )}
      </div>
      <RoutineDialogs actions={actions} onSaved={job => selectRoutine(refOf(job))} source={source} />
    </Panel>
  )
}
