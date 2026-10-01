/** Swimlanes: horizontal rows that split every column by one task field.
 *  Pure — grouping, lane order, and the drop/create semantics of each
 *  dimension live here as one table so the grid, the drag handler, and the
 *  new-task dialog agree on what a lane means. */

import type { KanbanBoard, KanbanColumn, KanbanTask } from './types'

export const SWIMLANE_OPTIONS = ['none', 'project', 'assignee', 'tenant', 'priority'] as const

export type SwimlaneBy = (typeof SWIMLANE_OPTIONS)[number]

export type LaneDimension = Exclude<SwimlaneBy, 'none'>

/** Lane key of a task that has no value for the dimension. Every dimension
 *  normalizes "absent" ('' / null / undefined) to this one key. */
export const NO_LANE = ''

/** The fields a board PATCH may carry for a drop. */
export interface TaskPatch {
  assignee?: string
  priority?: number
  status?: string
}

/** Create-time fields that make a new task land in a lane. */
export interface LanePreset {
  assignee?: string
  priority?: number
  project_id?: string
  tenant?: string
}

interface Dimension {
  key: (task: KanbanTask) => string
  /** Order of two non-empty lane keys; `label` resolves display names. */
  compare: (a: string, b: string, label: (key: string) => string) => number
  /** Patch that moves a card INTO lane `key`; null = a drag can't change this
   *  field (the backend PATCH has no such field, or changing it would move the
   *  task's workspace). */
  patch: ((key: string) => TaskPatch) | null
  preset: (key: string) => LanePreset
}

const byLabel = (a: string, b: string, label: (key: string) => string) => label(a).localeCompare(label(b))

const DIMENSIONS: Record<LaneDimension, Dimension> = {
  // A task's project anchors its worktree under that project's repo at create
  // time; re-homing it is not a PATCH, so project lanes are read-only targets.
  project: {
    key: task => task.project_id || NO_LANE,
    compare: byLabel,
    patch: null,
    preset: key => (key ? { project_id: key } : {})
  },
  assignee: {
    key: task => task.assignee || NO_LANE,
    compare: byLabel,
    // '' unassigns (PATCH treats an empty assignee as clear).
    patch: key => ({ assignee: key }),
    preset: key => (key ? { assignee: key } : {})
  },
  tenant: {
    key: task => task.tenant || NO_LANE,
    compare: byLabel,
    patch: null,
    preset: key => (key ? { tenant: key } : {})
  },
  // Every task has a priority (default 0), so there is no empty lane; higher
  // runs first, so higher sorts first — the dispatcher's own order.
  priority: {
    key: task => String(task.priority ?? 0),
    compare: (a, b) => Number(b) - Number(a),
    patch: key => ({ priority: Number(key) }),
    preset: key => ({ priority: Number(key) })
  }
}

export const normalizeSwimlaneBy = (value: unknown): SwimlaneBy =>
  (SWIMLANE_OPTIONS as readonly unknown[]).includes(value) ? (value as SwimlaneBy) : 'none'

export const laneKey = (by: LaneDimension, task: KanbanTask): string => DIMENSIONS[by].key(task)

export const lanePreset = (by: LaneDimension, key: string): LanePreset => DIMENSIONS[by].preset(key)

/** Whether a card can be dragged from one lane into another. */
export const laneIsDropTarget = (by: LaneDimension): boolean => DIMENSIONS[by].patch !== null

export interface Swimlane {
  key: string
  /** Every board column, in board order, holding only this lane's tasks. */
  columns: KanbanColumn[]
  count: number
}

/** Split the board's columns into lanes. Only lanes with at least one task
 *  exist; in-column task order is preserved; the no-value lane sorts last. */
export function groupSwimlanes(
  columns: KanbanColumn[],
  by: LaneDimension,
  label: (key: string) => string = key => key
): Swimlane[] {
  const dim = DIMENSIONS[by]
  const lanes = new Map<string, Swimlane>()

  for (const [index, column] of columns.entries()) {
    for (const task of column.tasks) {
      const key = dim.key(task)
      let lane = lanes.get(key)

      if (!lane) {
        lane = { key, columns: columns.map(col => ({ name: col.name, tasks: [] })), count: 0 }
        lanes.set(key, lane)
      }

      lane.columns[index].tasks.push(task)
      lane.count += 1
    }
  }

  return [...lanes.values()].sort((a, b) =>
    a.key === NO_LANE ? 1 : b.key === NO_LANE ? -1 : dim.compare(a.key, b.key, label)
  )
}

/** The PATCH for dropping `task` into cell (`toLane`, `toStatus`). `{}` is a
 *  no-op (same cell); null means the drop is refused. Column locking is the
 *  caller's rule (it applies with or without lanes). */
export function dropPatch(by: SwimlaneBy, task: KanbanTask, toLane: null | string, toStatus: string): null | TaskPatch {
  const patch: TaskPatch = toStatus === task.status ? {} : { status: toStatus }

  if (by === 'none' || toLane === null || DIMENSIONS[by].key(task) === toLane) {
    return patch
  }

  const change = DIMENSIONS[by].patch

  // A claimed card can't be reassigned (the backend refuses until reclaimed).
  if (!change || (by === 'assignee' && task.status === 'running')) {
    return null
  }

  return { ...patch, ...change(toLane) }
}

/** Optimistic board edit for a patch: move columns on a status change, merge
 *  the lane fields in place. The follow-up refresh reconciles. */
export function applyPatch(board: KanbanBoard, id: string, patch: TaskPatch): KanbanBoard {
  let moved: KanbanTask | undefined

  const columns = board.columns.map(col => ({
    ...col,
    tasks: col.tasks.flatMap(task => {
      if (task.id !== id) {
        return [task]
      }

      moved = {
        ...task,
        ...(patch.status !== undefined && { status: patch.status }),
        ...(patch.assignee !== undefined && { assignee: patch.assignee || null }),
        ...(patch.priority !== undefined && { priority: patch.priority })
      }

      return patch.status === undefined ? [moved] : []
    })
  }))

  if (!moved) {
    return board
  }

  if (patch.status === undefined) {
    return { ...board, columns }
  }

  return {
    ...board,
    columns: columns.map(col => (col.name === patch.status ? { ...col, tasks: [moved!, ...col.tasks] } : col))
  }
}
