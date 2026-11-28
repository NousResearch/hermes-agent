/**
 * AgentTrace — local projections of the gateway wire contract.
 *
 * The plugin fence allows only `@hermes/plugin-sdk` imports, and the SDK does
 * not re-export the generated wire shapes, so these are hand-mirrored minimal
 * projections of `apps/shared/src/gateway-contract.generated.ts` (the Python
 * contracts in `tui_gateway/contracts` stay the source of truth). Every field
 * is optional where the emitter may omit it — older backends send less.
 */

/** `ToolStartPayload` — a tool call began (stable id + full args). */
export interface WireToolStart {
  tool_id: string
  name: string
  context?: string | null
  args?: Record<string, unknown> | null
  args_text?: string | null
  preview?: string | null
  labels?: WireToolLabel[] | null
}

/** `ToolCompletePayload` — a tool call finished (parsed result + duration). */
export interface WireToolComplete {
  tool_id: string
  name: string
  args?: Record<string, unknown> | null
  duration_s?: number | null
  result?: unknown
  summary?: string | null
  result_text?: string | null
  inline_diff?: string | null
  labels?: WireToolLabel[] | null
}

/** `ToolGeneratingPayload` — the model is emitting a call's arguments. */
export interface WireToolGenerating {
  name: string
}

/** `ToolProgressPayload` — live progress of a running call. */
export interface WireToolProgress {
  tool_id: string
  name: string
  progress: number
  total?: number | null
  percent?: number | null
  message?: string | null
}

/** `ToolLabel` — what a bridged call executed through, in words. */
export interface WireToolLabel {
  kind: string
  app: string
  action: string
  emoji: string
  text: string
  name: string
  preview?: string
}

/** `SubagentEventPayload` — every `subagent.*` frame. */
export interface WireSubagentEvent {
  goal: string
  task_count: number
  task_index: number
  subagent_id?: string | null
  parent_id?: string | null
  child_session_id?: string | null
  delegation_id?: string | null
  depth?: number | null
  model?: string | null
  tool_count?: number | null
  toolsets?: string[] | null
  input_tokens?: number | null
  output_tokens?: number | null
  reasoning_tokens?: number | null
  api_calls?: number | null
  files_read?: string[] | null
  files_written?: string[] | null
  output_tail?: WireOutputTailEntry[] | null
  tool_name?: string | null
  text?: string | null
  status?: WireSubagentStatus | null
  summary?: string | null
  duration_seconds?: number | null
  tool_preview?: string | null
}

export interface WireOutputTailEntry {
  tool: string
  preview: string
  is_error: boolean
}

/** `SubagentStatus` — terminal: failed / error / timeout / interrupted / completed. */
export type WireSubagentStatus = 'queued' | 'running' | 'completed' | 'failed' | 'error' | 'timeout' | 'interrupted'

/** `ActiveSubagent` — one live child from `delegation.status`. */
export interface WireActiveSubagent {
  subagent_id: string
  parent_id?: string | null
  depth?: number | null
  goal?: string | null
  delegation_id?: string | null
  model?: string | null
  started_at?: number | null
  status?: string | null
  tool_count?: number | null
  owner_agent_session_id?: string | null
  [key: string]: unknown
}

/** `DelegationStatusResult` — the running subagent tree + spawn limits. */
export interface WireDelegationStatus {
  active: WireActiveSubagent[]
  paused: boolean
  max_spawn_depth: number
  max_concurrent_children: number
}

/** `SubagentSnapshot` — one live child from `subagent.list`. */
export interface WireSubagentSnapshot {
  subagent_id: string
  parent_id?: string | null
  depth?: number | null
  goal?: string | null
  delegation_id?: string | null
  model?: string | null
  started_at?: number | null
  status?: WireSubagentStatus | null
  tool_count?: number | null
  last_tool?: string | null
  accepting_steer?: boolean | null
  child_session_id?: string | null
}

/** `FailedDelegation` — one failed async task from `subagent.list`. */
export interface WireFailedDelegation {
  delegation_id: string
  task_index?: number
  status: string
  goal?: string
  error?: string | null
  dispatched_at?: number | null
  completed_at?: number | null
}

/** `SubagentListResult`. */
export interface WireSubagentList {
  subagents?: WireSubagentSnapshot[]
  delegations?: WireFailedDelegation[]
}

/** `Usage` — the streamed status-bar usage block (`session.usage` events). */
export interface WireUsage {
  model?: string
  input?: number
  output?: number
  reasoning?: number
  prompt?: number
  completion?: number
  total?: number
  calls?: number
  compressions?: number | null
  context_used?: number | null
  context_max?: number | null
  context_percent?: number | null
  context_source?: string | null
  context_estimated?: boolean | null
  cache_hit_pct?: number | null
  cache_read?: number | null
  cache_write?: number | null
  avg_latency_s?: number | null
  avg_tps?: number | null
  active_subagents?: number | null
  cost_usd?: number | null
  [key: string]: unknown
}

/** `SessionUsagePayload` — `usage` is the modern block; older frames carry
 *  the deprecated top-level counters, which we fold into the same shape. */
export interface WireSessionUsage {
  usage?: WireUsage | null
  context_tokens?: number | null
  context_length?: number | null
  input_tokens?: number | null
  output_tokens?: number | null
  total_tokens?: number | null
  api_calls?: number | null
  avg_latency_s?: number | null
  tokens_per_s?: number | null
  active_subagents?: number | null
  model?: string | null
  provider?: string | null
  [key: string]: unknown
}

/** `TranscriptMessage` / `StoredTranscriptRow` tool-row projection — the two
 *  history doors (`session.history`, canonical `session.transcript`) carry
 *  the same tool fields, so one hydration path reads both. */
export interface WireHistoryRow {
  role: string
  text?: string | null
  content?: unknown | null
  tool_call_id?: string | null
  timestamp?: number | null
  row_id?: number | null
  name?: string | null
  context?: string | null
  args?: Record<string, unknown> | null
  labels?: WireToolLabel[] | null
  [key: string]: unknown
}

/** `SessionLiveInfo` — the `session.info` event (model/title/provider). */
export interface WireSessionInfo {
  model?: string
  provider?: string
  title?: string
  running?: boolean
  usage?: WireUsage | null
  [key: string]: unknown
}

/** `MessageCompletePayload` — turn end (usage + outcome). */
export interface WireMessageComplete {
  usage?: WireUsage | null
  status?: string | null
  error?: string | null
  [key: string]: unknown
}

// ── Trace model (what the UI renders) ───────────────────────────────────────

export type ToolCallStatus = 'generating' | 'running' | 'done' | 'error'

/** One tool call — live (`tool.*` events) or hydrated from history. */
export interface ToolCall {
  id: string
  name: string
  status: ToolCallStatus
  /** 'live' = observed events (full args/result); 'history' = persisted row
   *  (args + preview; output only survives for a few tools). */
  origin: 'live' | 'history'
  args?: Record<string, unknown> | null
  argsText?: string | null
  preview?: string | null
  labels?: WireToolLabel[] | null
  startedAt?: number | null
  endedAt?: number | null
  durationS?: number | null
  resultText?: string | null
  summary?: string | null
  error?: string | null
  resultChars?: number | null
  truncated?: boolean
  progress?: { value: number; total?: number | null; message?: string | null } | null
  /** Set for `subagent.tool` progress rows (name + preview only). */
  subagentId?: string | null
  /** The last raw frame (complete payload or history row), for the Raw tab. */
  raw?: unknown
}

export type SubagentNodeStatus = WireSubagentStatus | 'spawn_requested'

/** One delegated child — a node in the session's execution tree. */
export interface SubagentNode {
  id: string
  parentId?: string | null
  depth: number
  goal: string
  taskIndex: number
  taskCount: number
  model?: string | null
  status: SubagentNodeStatus
  startedAt?: number | null
  endedAt?: number | null
  durationS?: number | null
  toolCount: number
  lastTool?: string | null
  childSessionId?: string | null
  delegationId?: string | null
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  apiCalls: number
  outputTail: WireOutputTailEntry[]
  summary?: string | null
  /** Tool-name progress frames the child emitted (`subagent.tool`). */
  tools: ToolCall[]
  thinkingTail?: string | null
  error?: string | null
}

/** Everything traced for one runtime session. */
export interface SessionTrace {
  sessionId: string
  profile?: string | null
  connectionId?: string | null
  title?: string | null
  model?: string | null
  provider?: string | null
  firstSeenAt: number
  lastActivityAt: number
  turnActive: boolean
  usage?: WireUsage | null
  lastError?: { message: string; at: number } | null
  generating?: { name: string; startedAt: number } | null
  tools: ToolCall[]
  subagents: SubagentNode[]
  historyHydrated?: boolean
}

export interface TraceState {
  sessions: SessionTrace[]
}

/** What the inspection drawer is showing. */
export type SelectedStep =
  | { kind: 'tool'; sessionId: string; toolId: string; subagentId?: string | null }
  | { kind: 'subagent'; sessionId: string; subagentId: string }
  | { kind: 'session'; sessionId: string }

/** The normalized event envelope the reducer consumes (Unix seconds). */
export interface TraceEvent {
  type: string
  at: number
  sessionId?: string | null
  profile?: string | null
  connectionId?: string | null
  payload?: unknown
}

export const TERMINAL_SUBAGENT_STATUSES: readonly SubagentNodeStatus[] = [
  'completed',
  'failed',
  'error',
  'timeout',
  'interrupted'
]

export function isTerminalSubagent(status: SubagentNodeStatus): boolean {
  return TERMINAL_SUBAGENT_STATUSES.includes(status)
}
