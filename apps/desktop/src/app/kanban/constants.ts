import type { KanbanStatus } from '@/types/hermes'

// Column order mirrors the backend dashboard (`BOARD_COLUMNS` in the kanban
// plugin). "archived" is intentionally excluded — it is a filter, not a column.
export const KANBAN_COLUMNS: KanbanStatus[] = [
  'triage',
  'todo',
  'scheduled',
  'ready',
  'running',
  'blocked',
  'review',
  'done'
]

export const KANBAN_COLUMN_LABELS: Record<KanbanStatus, string> = {
  archived: 'Archived',
  blocked: 'Blocked',
  done: 'Done',
  ready: 'Ready',
  review: 'Review',
  running: 'Running',
  scheduled: 'Scheduled',
  todo: 'To do',
  triage: 'Triage'
}

const KANBAN_COLUMN_SET: ReadonlySet<string> = new Set(KANBAN_COLUMNS)

export function isKanbanColumn(value: string): value is KanbanStatus {
  return KANBAN_COLUMN_SET.has(value)
}
