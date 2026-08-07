import type { KanbanBoard, KanbanTask } from './types'

export interface KanbanBoardFilters {
  assignee: string
  search: string
  showCompletedChildren: boolean
  tenant: string
}

export interface FilteredKanbanBoard {
  board: KanbanBoard
  /** Completed child cards collapsible under a currently visible parent. */
  completedChildren: number
}

const matchesOrdinaryFilters = (task: KanbanTask, filters: KanbanBoardFilters, query: string) =>
  (!query || `${task.title} ${task.body ?? ''} ${task.id}`.toLowerCase().includes(query)) &&
  (!filters.tenant || task.tenant === filters.tenant) &&
  (!filters.assignee || task.assignee === filters.assignee)

/**
 * Keep workflow history without letting each completed step occupy the Done
 * lane forever. Parent/root cards stay visible and retain the backend's N/M
 * rollup; completed children remain discoverable through search, the parent
 * drawer, or the explicit filter-menu toggle.
 */
export function filterKanbanBoard(board: KanbanBoard, filters: KanbanBoardFilters): FilteredKanbanBoard {
  const query = filters.search.trim().toLowerCase()
  let completedChildren = 0

  const candidates = board.columns.map(column => ({
    ...column,
    tasks: column.tasks.filter(task => matchesOrdinaryFilters(task, filters, query))
  }))

  const visibleParentIds = new Set(candidates.flatMap(column => column.tasks.map(task => task.id)))

  const columns = candidates.map(column => {
    const tasks = column.tasks.filter(task => {
      const hasVisibleParent = task.collapse_parent_ids?.some(parentId => visibleParentIds.has(parentId)) ?? false

      if (task.status !== 'done' || !hasVisibleParent) {
        return true
      }

      completedChildren += 1

      // An explicit search is already a request to find matching history, so
      // never make the user toggle a second control to open the result.
      return filters.showCompletedChildren || Boolean(query)
    })

    return { ...column, tasks }
  })

  return { board: { ...board, columns }, completedChildren }
}

/** Drop hidden cards from bulk selection so the action count stays truthful. */
export function pruneTaskSelection(selected: ReadonlySet<string>, board: KanbanBoard): ReadonlySet<string> {
  const visible = new Set(board.columns.flatMap(column => column.tasks.map(task => task.id)))
  const kept = [...selected].filter(id => visible.has(id))

  return kept.length === selected.size ? selected : new Set(kept)
}
