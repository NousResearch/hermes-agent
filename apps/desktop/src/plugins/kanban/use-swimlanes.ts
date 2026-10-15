/** The board page's swimlane wiring: the chosen grouping, project names, the
 *  lanes themselves, and the drop / "+" handlers a lane cell needs. Kept out
 *  of KanbanBoardPage so the page composes and this owns the lane policy. */

import { host, useQuery, useValue } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

import { $swimlaneBy, fetchProjects, projectsKey, useKanbanScope } from './api'
import { laneTitle } from './swimlane-grid'
import {
  dropPatch,
  groupSwimlanes,
  type LaneDimension,
  type LanePreset,
  lanePreset,
  normalizeSwimlaneBy,
  type TaskPatch
} from './swimlanes'
import type { KanbanColumn, KanbanTask } from './types'
import { isLockedTarget, lockedReason, useKanban } from './ui'

/** A new-task request: the column it lands in, plus — when raised from a
 *  swimlane cell — the lane's create-time fields and display name. */
export interface NewTaskRequest {
  status: string
  preset?: LanePreset
  laneTitle?: string
  /** The lane the task is meant for, to say so if it lands elsewhere. */
  lane?: { by: LaneDimension; key: string; title: (key: string) => string }
}

export function useSwimlanes({
  columns,
  findTask,
  move
}: {
  columns: KanbanColumn[] | null
  findTask: (id: string) => KanbanTask | undefined
  move: (id: string, patch: TaskPatch) => void
}) {
  const k = useKanban()
  const scope = useKanbanScope()
  const swimlaneBy = normalizeSwimlaneBy(useValue($swimlaneBy))
  const by = swimlaneBy === 'none' ? null : swimlaneBy
  // Names for project lanes; the board switcher reads the same cached query.
  const { data: projectList } = useQuery({ queryKey: projectsKey(scope), queryFn: fetchProjects, staleTime: 30_000 })

  const projectNames = useMemo(
    () => new Map((projectList?.projects ?? []).map(project => [project.id, project.name])),
    [projectList]
  )

  const lanes = useMemo(() => {
    if (!by || !columns) {
      return null
    }

    // Every live project gets a lane, so the first task can be created into it.
    const seed = by === 'project' ? [...projectNames.keys()] : []

    return groupSwimlanes(columns, by, key => laneTitle(by, key, k, projectNames), seed)
  }, [columns, by, k, projectNames])

  /** The patch for a drop, or null when it is refused (lane rule or a locked column). */
  const patchFor = (id: string, lane: null | string, status: string): null | TaskPatch => {
    const task = findTask(id)
    const patch = task ? dropPatch(swimlaneBy, task, lane, status) : null

    return patch && !(patch.status && isLockedTarget(patch.status)) ? patch : null
  }

  // One drop rule for the flat board (lane = null: status only) and the
  // swimlane grid (a cross-lane drop also writes the lane's field).
  const drop = (id: string, lane: null | string, status: string) => {
    const patch = patchFor(id, lane, status)

    if (patch && Object.keys(patch).length > 0) {
      move(id, patch)
    } else if (isLockedTarget(status) && findTask(id)?.status !== status) {
      host.notify({ kind: 'info', message: lockedReason(k, status) })
    }
  }

  const newTaskIn = (status: string, lane: string): NewTaskRequest => {
    if (!by) {
      return { status }
    }

    const title = (key: string) => laneTitle(by, key, k, projectNames)

    return { status, preset: lanePreset(by, lane), laneTitle: title(lane), lane: { by, key: lane, title } }
  }

  return {
    by,
    lanes,
    projectNames,
    canDrop: (id: string, lane: string, status: string) => patchFor(id, lane, status) !== null,
    drop,
    newTaskIn
  }
}
