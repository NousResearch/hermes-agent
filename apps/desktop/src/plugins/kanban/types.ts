/** The slice of the kanban REST contract the board renders. The backend
 *  (`plugins/kanban/dashboard/plugin_api.py`) returns much more per task; we
 *  type only what the UI reads so a schema addition never breaks the build. */

/** One card. `status` is the column id (see COLUMN_META). */
export interface KanbanTask {
  id: string
  title: string
  body?: null | string
  status: string
  assignee?: null | string
  priority?: number
  tenant?: null | string
  created_at?: number
  latest_summary?: null | string
  comment_count?: number
  link_counts?: { parents: number; children: number }
  /** N-of-M child completion, or null when the task has no children. */
  progress?: null | { done: number; total: number }
  /** Compact diagnostics rollup — present only when a card has warnings. */
  warnings?: null | { count: number; highest_severity?: null | string }
  /** Worker liveness (present on running cards) — drives the arc + run clock. */
  started_at?: null | number
  worker_pid?: null | number
  last_heartbeat_at?: null | number
}

export interface KanbanColumn {
  name: string
  tasks: KanbanTask[]
}

export interface KanbanBoard {
  columns: KanbanColumn[]
  tenants: string[]
  assignees: string[]
  latest_event_id: number
  now: number
}

/** A structured recovery action attached to a diagnostic. */
export interface DiagnosticAction {
  kind: string
  label: string
  payload?: Record<string, unknown>
  suggested?: boolean
}

/** One active distress signal on a task (kanban_diagnostics.Diagnostic). */
export interface Diagnostic {
  kind: string
  severity: 'critical' | 'error' | 'warning'
  title: string
  detail: string
  actions: DiagnosticAction[]
  count: number
  last_seen_at: number
  data: Record<string, unknown>
}

export interface KanbanRun {
  id: number | string
  profile?: null | string
  status: string
  outcome?: null | string
  summary?: null | string
  error?: null | string
  metadata?: null | Record<string, unknown> | string
  worker_pid?: null | number
  started_at?: null | number
  ended_at?: null | number
}

export interface KanbanComment {
  id: number | string
  author: string
  body: string
  created_at: number
}

export interface KanbanEvent {
  id: number
  kind: string
  payload: unknown
  created_at: number
}

export interface KanbanAttachment {
  id: number | string
  filename: string
  size?: null | number
}

/** Fields present only on the detail endpoint (beyond the card's KanbanTask).
 *  `started_at`/`worker_pid`/`last_heartbeat_at` are inherited — they live on
 *  KanbanTask now that the board's liveness arc reads them. */
export interface KanbanTaskFull extends KanbanTask {
  result?: null | string
  created_by?: null | string
  completed_at?: null | number
  last_failure_error?: null | string
  workspace_kind?: null | string
  workspace_path?: null | string
  branch_name?: null | string
  consecutive_failures?: number
  diagnostics?: Diagnostic[]
}

/** GET /tasks/:id — the task plus its related collections, which are SIBLINGS
 *  of `task`, not nested inside it. */
export interface KanbanTaskDetail {
  task: KanbanTaskFull
  comments: KanbanComment[]
  events: KanbanEvent[]
  attachments: KanbanAttachment[]
  links: { parents: string[]; children: string[] }
  runs: KanbanRun[]
}

/** GET /boards — every board on disk + which one is the server's current. */
export interface BoardMeta {
  slug: string
  name?: null | string
  description?: null | string
  is_current?: boolean
  total?: number
  /** Board-level project directory new tasks inherit (empty = none). */
  default_workdir?: null | string
  /** Recommended workspace kind derived from default_workdir by the backend
   *  (`scratch` when unset, `worktree` in a git repo, else `dir`). */
  default_workspace_kind?: null | string
  /** First-class Project the board is scoped to (id) + resolved name. */
  project_id?: null | string
  project_name?: null | string
}

/** GET /projects — first-class Hermes projects available to scope a board. */
export interface KanbanProject {
  id: string
  slug: string
  name: string
  primary_path?: null | string
  icon?: null | string
  color?: null | string
}

/** POST /tasks/:id/estimate — rough auxiliary-model estimate (never dollars). */
export interface TaskEstimate {
  ok: boolean
  reason?: null | string
  est_tokens?: number
  complexity?: 'L' | 'M' | 'S' | null
  rationale?: null | string
  model?: null | string
}
export interface BoardsResponse {
  boards: BoardMeta[]
  current: string
}

/** GET /tasks/:id/log — the worker's stdout/stderr tail. */
export interface WorkerLog {
  exists: boolean
  size_bytes: number
  content: string
  truncated: boolean
}

/** GET /orchestration — dispatcher knobs from config.yaml + resolved values. */
export interface OrchestrationSettings {
  orchestrator_profile: string
  default_assignee: string
  auto_decompose: boolean
  resolved_orchestrator_profile: string
  resolved_default_assignee: string
}

/** GET /profiles — the roster the decomposer routes across. */
export interface KanbanProfile {
  name: string
  is_default: boolean
  description: string
  description_auto: boolean
}

/** Column presentation — codicon + tone only. Labels + help live in i18n
 *  (plugin bundles); see `columnLabel`/`columnHelp` in i18n.ts. Order follows
 *  the backend's BOARD_COLUMNS; anything the backend adds renders via the
 *  fallback. */
export const COLUMN_META: Record<string, { codicon: string; tone: string }> = {
  triage: { codicon: 'inbox', tone: 'var(--ui-text-tertiary)' },
  todo: { codicon: 'circle-outline', tone: 'var(--ui-text-secondary)' },
  scheduled: { codicon: 'watch', tone: '#a78bfa' },
  ready: { codicon: 'play-circle', tone: '#60a5fa' },
  running: { codicon: 'sync', tone: '#34d399' },
  blocked: { codicon: 'error', tone: '#f87171' },
  review: { codicon: 'eye', tone: '#fbbf24' },
  done: { codicon: 'pass', tone: 'var(--ui-text-tertiary)' },
  archived: { codicon: 'archive', tone: 'var(--ui-text-quaternary)' }
}

export const columnMeta = (name: string) =>
  COLUMN_META[name] ?? { codicon: 'circle-outline', tone: 'var(--ui-text-secondary)' }

export const SEVERITY_TONE: Record<Diagnostic['severity'], string> = {
  critical: 'var(--destructive, #f87171)',
  error: 'var(--destructive, #f87171)',
  warning: '#fbbf24'
}
