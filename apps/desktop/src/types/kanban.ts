// Response shapes for the kanban dashboard plugin REST API
// (plugins/kanban/dashboard/plugin_api.py, mounted at /api/plugins/kanban/).
// Kanban is HOME-scoped: boards are shared across profiles by design, so none
// of these payloads carry a profile dimension.

export interface KanbanTaskAge {
  created_age_seconds: null | number
  started_age_seconds: null | number
  time_to_complete_seconds: null | number
}

export interface KanbanDiagnosticAction {
  kind: string
  label: string
  payload: Record<string, unknown>
  suggested: boolean
}

export interface KanbanDiagnostic {
  kind: string
  severity: 'critical' | 'error' | 'warning' | (string & {})
  title: string
  detail: string
  actions: KanbanDiagnosticAction[]
  first_seen_at: number
  last_seen_at: number
  count: number
}

export interface KanbanWarningsSummary {
  count: number
  kinds: Record<string, number>
  latest_at: number
  highest_severity: null | string
}

export interface KanbanTask {
  id: string
  title: string
  body: null | string
  assignee: null | string
  status: string
  priority: number
  created_by: null | string
  created_at: number
  started_at: null | number
  completed_at: null | number
  workspace_kind: string
  workspace_path: null | string
  claim_lock: null | string
  claim_expires: null | number
  tenant: null | string
  branch_name?: null | string
  project_id?: null | string
  result?: null | string
  idempotency_key?: null | string
  consecutive_failures?: number
  worker_pid?: null | number
  last_failure_error?: null | string
  max_runtime_seconds?: null | number
  last_heartbeat_at?: null | number
  current_run_id?: null | number
  workflow_template_id?: null | string
  current_step_key?: null | string
  skills?: null | string[]
  model_override?: null | string
  session_id?: null | string
  age?: KanbanTaskAge
  /** Latest non-null run summary. 200-char preview on /board, full on /tasks/:id. */
  latest_summary?: null | string
  diagnostics?: KanbanDiagnostic[]
  warnings?: KanbanWarningsSummary | null
}

/** Board-card projection: a task plus the per-card rollups /board attaches. */
export interface KanbanCard extends KanbanTask {
  link_counts?: { parents: number; children: number }
  comment_count?: number
  /** Children progress rollup; null/absent when the task has no children. */
  progress?: { done: number; total: number } | null
}

export interface KanbanColumn {
  name: string
  tasks: KanbanCard[]
}

export interface KanbanBoardPayload {
  columns: KanbanColumn[]
  tenants: string[]
  assignees: string[]
  /** Max task_events id — cheap change cursor for polling. */
  latest_event_id: number
  now: number
}

export interface KanbanComment {
  id: number
  task_id: string
  author: null | string
  body: string
  created_at: number
}

export interface KanbanEvent {
  id: number
  task_id: string
  kind: string
  payload: null | Record<string, unknown>
  created_at: number
  run_id: null | number
}

export interface KanbanAttachment {
  id: number
  task_id: string
  filename: string
  content_type: null | string
  size: null | number
  uploaded_by: null | string
  stored_path: string
  created_at: number
}

export interface KanbanRun {
  id: number
  task_id: string
  profile: null | string
  step_key: null | string
  status: string
  claim_lock: null | string
  claim_expires: null | number
  worker_pid: null | number
  max_runtime_seconds: null | number
  last_heartbeat_at: null | number
  started_at: number
  ended_at: null | number
  outcome: null | string
  summary: null | string
  metadata: null | Record<string, unknown>
  error: null | string
}

export interface KanbanLinks {
  parents: string[]
  children: string[]
}

export interface KanbanTaskDetail {
  task: KanbanTask
  comments: KanbanComment[]
  events: KanbanEvent[]
  attachments: KanbanAttachment[]
  links: KanbanLinks
  runs: KanbanRun[]
}

export interface KanbanBoardInfo {
  slug: string
  name: string
  description: string
  icon: string
  color: string
  default_workdir?: null | string
  created_at?: null | number
  archived?: boolean
  is_current?: boolean
  counts?: Record<string, number>
  total?: number
}

export interface KanbanBoardsResponse {
  boards: KanbanBoardInfo[]
  current: string
}

export interface KanbanAssignee {
  name: string
  on_disk: boolean
  counts: Record<string, number>
}

export interface KanbanCreateTaskPayload {
  title: string
  body?: string
  assignee?: string
  tenant?: string
  priority?: number
  parents?: string[]
  triage?: boolean
}

export interface KanbanCreateTaskResponse {
  task: KanbanTask | null
  /** Dispatcher-presence warning (ready+assigned task with no dispatcher running). */
  warning?: string
}

export interface KanbanUpdateTaskPayload {
  status?: string
  assignee?: string
  priority?: number
  title?: string
  body?: string
  result?: string
  block_reason?: string
  summary?: string
}

export interface KanbanUpdateTaskResponse {
  task: KanbanTask | null
}

export interface KanbanDispatchResult {
  reclaimed?: number
  promoted?: number
  spawned?: [string, string, string][]
  skipped_unassigned?: string[]
  auto_assigned_default?: string[]
}

export interface KanbanTaskLog {
  task_id: string
  path: string
  exists: boolean
  size_bytes: number
  content: string
  truncated: boolean
}
