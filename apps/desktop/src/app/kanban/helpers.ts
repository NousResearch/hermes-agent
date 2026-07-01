import type { KanbanBoardResponse, KanbanStatus, KanbanTask } from '@/types/hermes'

const SECONDS_PER_MINUTE = 60
const SECONDS_PER_HOUR = 3600
const SECONDS_PER_DAY = 86_400

// Compact relative age ("3m", "2h", "5d") from a unix-second timestamp, using
// the board's server `now` so client clock skew never shows negative ages.
export function formatAge(createdAt: number | null | undefined, now: number): string {
  if (!createdAt) {
    return ''
  }

  const delta = Math.max(0, now - createdAt)

  if (delta < SECONDS_PER_MINUTE) {
    return `${delta}s`
  }

  if (delta < SECONDS_PER_HOUR) {
    return `${Math.floor(delta / SECONDS_PER_MINUTE)}m`
  }

  if (delta < SECONDS_PER_DAY) {
    return `${Math.floor(delta / SECONDS_PER_HOUR)}h`
  }

  return `${Math.floor(delta / SECONDS_PER_DAY)}d`
}

function matchesQuery(task: KanbanTask, query: string): boolean {
  if (!query) {
    return true
  }

  const haystack = `${task.title} ${task.assignee ?? ''} ${task.body ?? ''}`.toLowerCase()

  return haystack.includes(query)
}

function matchesAssignee(task: KanbanTask, assignee: string | null): boolean {
  if (!assignee) {
    return true
  }

  return (task.assignee ?? '') === assignee
}

export interface FilterBoardArgs {
  board: KanbanBoardResponse
  query: string
  assignee: string | null
}

// Returns a status→tasks map filtered by the free-text query and the active
// assignee, preserving the backend column ordering within each status.
export function filterTasksByStatus(args: FilterBoardArgs): Record<KanbanStatus, KanbanTask[]> {
  const q = args.query.trim().toLowerCase()
  const grouped = {} as Record<KanbanStatus, KanbanTask[]>

  args.board.columns.forEach(column => {
    grouped[column.name] = column.tasks.filter(
      task => matchesQuery(task, q) && matchesAssignee(task, args.assignee)
    )
  })

  return grouped
}

export interface MoveTaskArgs {
  board: KanbanBoardResponse
  taskId: string
  toStatus: KanbanStatus
}

// Immutably relocates a task to another status column for optimistic rendering
// while the PATCH is in flight; the next poll reconciles against server truth.
export function moveTaskStatus(args: MoveTaskArgs): KanbanBoardResponse {
  const moved = args.board.columns.flatMap(column => column.tasks).find(task => task.id === args.taskId)

  if (!moved || moved.status === args.toStatus) {
    return args.board
  }

  const columns = args.board.columns.map(column => {
    if (column.name === moved.status) {
      return { ...column, tasks: column.tasks.filter(task => task.id !== args.taskId) }
    }

    if (column.name === args.toStatus) {
      return { ...column, tasks: [{ ...moved, status: args.toStatus }, ...column.tasks] }
    }

    return column
  })

  return { ...args.board, columns }
}
