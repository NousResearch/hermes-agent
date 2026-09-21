// Control Room contract — TypeScript mirror (generated, do not edit).
// Source of truth: control_room/contract.py (Python, pydantic v2).
// Regenerate with: python -m control_room.export_ts_schema
// Wire format is snake_case; these types mirror the JSON payloads exactly.

export type AttentionSeverity = 0 | 1 | 2 | 3;

export type AttentionKind = "approval" | "held_message" | "error" | "stalled" | "blocked_task" | "review_task" | "running" | "ready" | "system" | "info";

export type ErrorCode = "snapshot_unavailable" | "stale_target" | "cross_profile" | "unauthorized" | "unknown_action" | "unknown_target" | "backend_failed" | "confirmation_required";

export interface SourceMeta {
  provider: string
  state?: "ok" | "degraded" | "unavailable"
  detail?: string
}

export interface AttentionItem {
  kind: AttentionKind
  id: string
  severity: AttentionSeverity
  title: string
  detail?: string
  profile?: string
  source?: SourceMeta
  updated_at?: string
  available_actions?: Array<string>
}

export interface AgentRow {
  kind?: "agent" | "delegation" | "process"
  id: string
  name: string
  status?: string
  detail?: string
  profile?: string
  source?: SourceMeta
  available_actions?: Array<string>
}

export interface TaskRow {
  id: string
  title: string
  state?: string
  board?: string
  owner?: string
  profile?: string
  latest_run?: string
  source?: SourceMeta
  available_actions?: Array<string>
}

export interface MessageRow {
  kind?: "peer_message" | "peer_request"
  id: string
  title: string
  state?: string
  sender?: string
  profile?: string
  source?: SourceMeta
  available_actions?: Array<string>
}

export interface SystemSummary {
  state?: string
  severity?: AttentionSeverity
  detail?: string
  errors?: Array<string>
  source?: SourceMeta
  available_actions?: Array<string>
}

export interface Capabilities {
  approvals?: boolean
  peer_messages?: boolean
  kanban_actions?: boolean
  process_control?: boolean
  delegation_control?: boolean
}

export interface SnapshotCounts {
  needs_you?: number
  agents_active?: number
  tasks_running?: number
  messages_unread?: number
  system_severity?: AttentionSeverity
}

export interface ActionTarget {
  kind: string
  id: string
  profile?: string | null
}

export interface ControlRoomAction {
  id: string
  target: ActionTarget
  parameters?: Record<string, unknown>
  confirmation?: "none" | "required"
  expected_revision?: string | null
}

export interface ControlRoomActionResult {
  status?: "completed" | "confirmation_required" | "rejected" | "stale" | "unavailable" | "failed"
  receipt?: Record<string, unknown> | null
  message?: string
}

export interface ControlRoomError {
  code: ErrorCode
  message: string
  details?: Record<string, unknown>
}

export interface ControlRoomSnapshot {
  version?: number
  profile?: string
  generated_at?: string
  attention?: Array<AttentionItem>
  counts?: SnapshotCounts
  agents?: Array<AgentRow>
  tasks?: Array<TaskRow>
  messages?: Array<MessageRow>
  system?: SystemSummary
  capabilities?: Capabilities
}
