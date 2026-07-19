/**
 * Native Desktop client for the kanban plugin REST API
 * (`/api/plugins/kanban/*`) via `pluginRest` — no web dashboard SPA.
 */
import { pluginRest } from '@/hermes'

export const KANBAN_PLUGIN_ID = 'kanban'

/** Board columns left→right (matches plugins/kanban/dashboard/plugin_api.py). */
export const KANBAN_COLUMNS = [
  'triage',
  'todo',
  'scheduled',
  'ready',
  'running',
  'blocked',
  'review',
  'done'
] as const

export type KanbanStatus = (typeof KANBAN_COLUMNS)[number]

export interface KanbanTask {
  id: string
  title: string
  body?: string | null
  status: string
  assignee?: string | null
  tenant?: string | null
  priority?: number
  comment_count?: number
  age?: string | number | null
  latest_summary?: string | null
  link_counts?: { parents: number; children: number }
  progress?: { done: number; total: number } | null
}

export interface KanbanColumn {
  name: string
  tasks: KanbanTask[]
}

export interface KanbanBoard {
  board?: string
  columns: KanbanColumn[]
  tenants?: string[]
  assignees?: string[]
  latest_event_id?: number
}

export interface KanbanTaskDetail {
  task: KanbanTask
  comments?: Array<{ id?: number; author?: string; body?: string; created_at?: string }>
  events?: Array<{ id?: number; kind?: string; message?: string; created_at?: string }>
}

function qs(board?: string | null): string {
  if (!board) return ''
  return `?board=${encodeURIComponent(board)}`
}

export async function fetchKanbanBoard(board?: string | null): Promise<KanbanBoard> {
  return pluginRest<KanbanBoard>(KANBAN_PLUGIN_ID, `/board${qs(board)}`)
}

export async function fetchKanbanTask(taskId: string, board?: string | null): Promise<KanbanTaskDetail> {
  return pluginRest<KanbanTaskDetail>(KANBAN_PLUGIN_ID, `/tasks/${encodeURIComponent(taskId)}${qs(board)}`)
}

export async function createKanbanTask(
  input: {
    title: string
    body?: string
    assignee?: string
    priority?: number
    triage?: boolean
  },
  board?: string | null
): Promise<{ task: KanbanTask | null; warning?: string }> {
  return pluginRest(KANBAN_PLUGIN_ID, `/tasks${qs(board)}`, {
    method: 'POST',
    body: input
  })
}

export async function updateKanbanTask(
  taskId: string,
  patch: {
    status?: string
    assignee?: string | null
    priority?: number
    title?: string
    body?: string
    block_reason?: string
    summary?: string
  },
  board?: string | null
): Promise<{ task?: KanbanTask }> {
  return pluginRest(KANBAN_PLUGIN_ID, `/tasks/${encodeURIComponent(taskId)}${qs(board)}`, {
    method: 'PATCH',
    body: patch
  })
}

/** Flatten board columns into a status → tasks map (missing cols → []). */
export function columnMap(board: KanbanBoard | null | undefined): Record<string, KanbanTask[]> {
  const map: Record<string, KanbanTask[]> = {}
  for (const name of KANBAN_COLUMNS) {
    map[name] = []
  }
  if (!board?.columns) {
    return map
  }
  for (const col of board.columns) {
    map[col.name] = Array.isArray(col.tasks) ? col.tasks : []
  }
  return map
}

export function statusLabel(status: string): string {
  if (!status) return ''
  return status.charAt(0).toUpperCase() + status.slice(1)
}
