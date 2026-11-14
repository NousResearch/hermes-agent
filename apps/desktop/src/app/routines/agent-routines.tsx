import './routines.css'

import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import type { AgentSettingsOwner } from '../agent-settings/data'
import { openSession } from '../open-session'

import { RoutineDetail } from './routine-detail'
import { RoutineDialogs } from './routine-dialogs'
import { useAgentRoutineSource } from './routine-source'
import { RoutinesOverview } from './routines-overview'
import { openRoutineEditor, type RoutineRef, sameRoutine } from './routines-store'
import { useRoutineActions } from './use-routine-actions'

/** Agent › Settings › Routines in the redesign: the rail page's list filtered to this agent, with the
 *  same detail and the same editor (agent fixed). Classic keeps AgentSchedulesSettings. */
export function AgentRoutines({ owner }: { owner: AgentSettingsOwner }) {
  const navigate = useNavigate()
  const location = useLocation()
  const source = useAgentRoutineSource(owner)
  const actions = useRoutineActions(source)
  const params = new URLSearchParams(location.search)
  const linked = params.get('routine')
  // The rail page links here (its "Edit in <agent> settings"): `routine` opens it, `edit` its editor.
  const [selection, setSelection] = useState<null | RoutineRef>(() => (linked ? { id: linked, profile: owner.profile } : null))
  const selected = selection ? (source.jobs.find(job => sameRoutine(job, selection)) ?? null) : null
  const editRequested = params.get('edit') === '1'

  useEffect(() => {
    if (!editRequested || !selected) {
      return
    }

    openRoutineEditor({ job: selected, mode: 'edit' })
    const next = new URLSearchParams(location.search)
    next.delete('edit')
    navigate({ pathname: location.pathname, search: `?${next}` }, { replace: true })
  }, [editRequested, location.pathname, location.search, navigate, selected])
  const create = () => openRoutineEditor({ lockProfile: true, mode: 'pick', profile: owner.profile })

  return (
    <>
      {selected ? (
        <RoutineDetail
          actions={actions}
          embedded
          job={selected}
          onBack={() => setSelection(null)}
          onOpenSession={id =>
            openSession(id, navigate, 'in-place', {
              ownerRoute: { connectionId: owner.connectionId ?? 'local', profile: owner.profile },
              workspaceMode: 'sessions'
            })
          }
          source={source}
        />
      ) : (
        <RoutinesOverview
          actions={actions}
          agentScoped
          headerCreate={false}
          onCreate={create}
          onOpen={job => setSelection({ id: job.id, profile: job.profile ?? owner.profile })}
          source={source}
        />
      )}
      <RoutineDialogs
        actions={actions}
        onSaved={job => setSelection({ id: job.id, profile: job.profile ?? owner.profile })}
        source={source}
      />
    </>
  )
}
