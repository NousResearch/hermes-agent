/**
 * Wire types for `tui_gateway` JSON-RPC notifications and the RPC responses the
 * TypeScript surfaces (Ink TUI, Desktop, web dashboard) share.
 *
 * Every notification arrives as `{jsonrpc: '2.0', method: 'event', params: GatewayEvent}`
 * (`tui_gateway/server.py::_event_frame`). `GatewayEventMap` is the single map from
 * event `type` to payload shape; `BACKEND_EVENT_NAMES` mirrors the emitter side and is
 * pinned to `gateway-events.json` by `gateway-events.test.ts` (vitest) and
 * `tests/tui_gateway/test_gateway_event_contract.py` (Python), so a name added on one
 * side without the other fails a test instead of drifting silently.
 *
 * Payload interfaces are typed from the Python emitters (file::symbol noted per
 * interface). Events whose payload no TS client reads yet are `Record<string, unknown>`;
 * they still MUST be keys so `on('x', …)` stays exhaustive.
 */
import type { BillingBlock } from './billing-types.js'
import type { HermesSkin } from './skin.js'

// ── Shared value shapes ──────────────────────────────────────────────

/** `tui_gateway/server.py::_get_usage` — a session's token/cost counters. */
export interface Usage {
  active_subagents?: number
  /** Rolling mean API latency over the last 10 calls (seconds). */
  avg_latency_s?: number
  /** Rolling output tokens/sec over the last 10 calls. */
  avg_tps?: number
  /** Session prompt-cache hit ratio (cache_read / prompt tokens, %). Omitted (not 0)
   *  when the provider reports no cache reads. */
  cache_hit_pct?: number
  cache_read?: number
  cache_write?: number
  calls: number
  compressions?: number
  context_max?: number
  context_percent?: number
  context_estimated?: boolean
  context_source?: string
  context_used?: number
  cost_status?: string
  cost_usd?: number
  dev_credits_spent_micros?: number
  input: number
  output: number
  reasoning?: number
  total: number
}

/** Advisory `{layer, code, retryable}` descriptor (`agent/error_surface.py`). */
export interface ErrorSurface {
  code?: string
  layer?: string
  retryable?: boolean
}

/** `tui_gateway/tool_progress.py::_normalize_todo_state` — full task snapshot. */
export interface TodoStatePayload {
  revision?: number
  todos?: unknown[]
}

export type SubagentStatus = 'completed' | 'error' | 'failed' | 'interrupted' | 'queued' | 'running' | 'timeout'

/** `tui_gateway/tool_progress.py::_progress_subagent` — every `subagent.*` frame. */
export interface SubagentEventPayload {
  api_calls?: number
  /** The child's own gateway session id — the key a watch window mirrors. */
  child_session_id?: string
  /** Batch (delegation) id this subagent belongs to — distinguishes
   *  interleaved `[n/N]` progress from concurrent or nested fan-outs. */
  delegation_id?: string
  depth?: number
  duration_seconds?: number
  files_read?: string[]
  files_written?: string[]
  goal: string
  input_tokens?: number
  model?: string
  output_tail?: { is_error?: boolean; preview?: string; tool?: string }[]
  output_tokens?: number
  parent_id?: null | string
  reasoning_tokens?: number
  status?: SubagentStatus
  subagent_id?: string
  summary?: string
  task_count?: number
  task_index: number
  text?: string
  tool_count?: number
  tool_name?: string
  tool_preview?: string
  toolsets?: string[]
}

// ── Event payloads ───────────────────────────────────────────────────

/** `tui_gateway/entry.py` (stdio) / `tui_gateway/ws.py` (WebSocket) first frame. */
export interface GatewayReadyPayload {
  /** Backends with the change watcher broadcast `*.changed` events; consumers
   *  demote their legacy polls to slow backstops. */
  change_events?: boolean
  /** WebSocket transport only: the server answers heartbeat pings. */
  heartbeat?: boolean
  /** Opaque token for this server process's `seq` numbering; a new epoch means
   *  replay watermarks must be discarded. */
  replay_epoch?: string
  skin?: HermesSkin
}

/** `tui_gateway/prompt_turn.py::_complete_turn_payload` and
 *  `tui_gateway/session_auto_continue.py::_emit_terminal_turn_error`. */
export interface MessageCompletePayload {
  /** Structured billing wall when the turn failed with FailoverReason.billing. */
  billing?: BillingBlock
  /** `status: "error"` — the failure message (`text` may be streamed output). */
  error?: string
  error_surface?: ErrorSurface
  failure_reason?: string | null
  /** `status: "error"` — `text` is streamed partial output to keep, not the error string. */
  partial?: boolean
  reasoning?: string
  /** `status: "error"` — the failed turn was retained and replays via `session.resume.inflight`. */
  recoverable?: boolean
  rendered?: string
  /** The final text was already previewed via `message.interim`; settle, don't duplicate. */
  response_previewed?: boolean
  status?: 'complete' | 'error' | 'interrupted' | string
  text?: string
  usage?: Usage
  /** History-commit note (e.g. a mid-turn desync the gateway surfaced instead of dropping). */
  warning?: string
}

/** `tui_gateway/tool_progress.py::_on_tool_start`. */
export interface ToolStartPayload {
  /** Full tool arguments — the 80-char `context` preview is display-only. */
  args?: Record<string, unknown>
  /** Verbose mode only: pretty-printed args. */
  args_text?: string
  context?: string
  name?: string
  /** Mirrored child tool rows carry a short preview instead of args. */
  preview?: string
  tool_id: string
  /** Not on the wire (`_on_tool_start` never sets it): the todo snapshot rides `tool.complete` /
   *  `todo.updated`. Kept because the TUI handler reads it and its fixtures exercise that path. */
  todos?: unknown[]
}

/** `tui_gateway/tool_progress.py::_on_tool_complete`. */
export interface ToolCompletePayload {
  args?: Record<string, unknown>
  duration_s?: number
  inline_diff?: string
  name?: string
  /** Parsed JSON when the tool returned JSON, else the raw string. */
  result?: unknown
  /** Verbose mode only. */
  result_text?: string
  revision?: number
  summary?: string
  tool_id: string
  todos?: unknown[]
}

export interface ToolGeneratingPayload {
  name?: string
}

/** `tui_gateway/tool_progress.py::_progress_output_risk`. */
export interface ToolOutputRiskPayload {
  findings?: string[]
  name?: string
  redacted?: boolean
  risk?: string
  tool_id?: string
}

export interface StatusUpdatePayload {
  kind?: string
  text?: string
}

export interface NotificationShowPayload {
  id?: string
  key?: string
  kind?: 'sticky' | 'ttl' | string
  level?: 'error' | 'info' | 'success' | 'warn' | string
  text?: string
  ttl_ms?: null | number
}

export interface NotificationClearPayload {
  key?: string
}

export interface TextPayload {
  text?: string
}

/** `message.delta` / `reasoning.delta` / `reasoning.available` / `thinking.delta`. */
export interface StreamDeltaPayload {
  rendered?: string
  text?: string
  /** Verbose reasoning mode is on for this session. */
  verbose?: boolean
}

export interface MessageInterimPayload {
  already_streamed?: boolean
  text: string
}

export interface SessionUsagePayload {
  usage?: Usage
}

export interface SessionTitlePayload {
  session_id?: string
  title?: string
}

/** `tui_gateway/methods_session.py` resume hydration progress. */
export interface SessionResumeProgressPayload {
  message?: string
  message_count?: number
  phase?: string
  status?: 'complete' | 'failed' | 'loading' | string
}

/** `tui_gateway/session_lifecycle.py::_announce_session_reclaimed`. */
export interface SessionReclaimedPayload {
  reason?: string
  session_id?: string
  stored_session_id?: string
}

export interface SessionControlUpdatePayload {
  control?: unknown
}

export interface ErrorPayload {
  message?: string
  reason?: string
}

export interface NoticePayload {
  message?: string
}

export interface ReactionPayload {
  kind?: string
}

export interface BillingStepUpVerificationPayload {
  user_code?: string
  verification_url: string
}

export interface VoiceStatusPayload {
  state?: 'idle' | 'listening' | 'transcribing' | string
}

export interface VoiceTranscriptPayload {
  no_speech_limit?: boolean
  stop_phrase?: boolean
  text?: string
  typed?: boolean
  voice_stopped?: boolean
}

export interface WakeDetectedPayload {
  phrase?: string
  profile?: null | string
  start_new_session?: boolean
}

export interface BrowserProgressPayload {
  level?: 'error' | 'info' | 'warn' | string
  message?: string
}

export interface MoaReferencePayload {
  count?: number
  index?: number
  label?: string
  text?: string
}

export interface MoaAggregatingPayload {
  aggregator?: string
}

export interface MoaProgressPayload {
  label?: string
  refs_done?: number
  refs_total?: number
}

export interface MoaPhasePayload {
  aggregator?: string
  phase?: string
  refs_done?: number
  refs_total?: number
}

// Blocking bridges (`tui_gateway/server.py::_block`): every `*.request` carries a
// `request_id`; the matching `*.expire` names the same id when the wait timed out.

export interface RequestExpirePayload {
  request_id: string
}

export interface ClarifyQuestion {
  choices?: null | string[]
  multi_select?: boolean
  qid: string
  question: string
}

export interface ClarifyRequestPayload {
  answers?: Record<string, string>
  choices?: null | string[]
  multi_select?: boolean
  question?: string
  questions?: ClarifyQuestion[]
  request_id: string
}

/** `tui_gateway/server.py::_approval_request_payload` (command redacted server-side). */
export interface ApprovalRequestPayload {
  allow_permanent?: boolean
  choices?: string[]
  command: string
  description: string
  request_id?: string
  smart_denied?: boolean
}

export interface SudoRequestPayload {
  request_id: string
}

export interface SecretRequestPayload {
  env_var: string
  prompt: string
  request_id: string
}

export interface VaultUnlockRequestPayload {
  backend: string
  display_name: string
  request_id: string
}

export interface VaultCodeRequestPayload {
  hint?: string
  request_id: string
  site?: string
}

export interface McpSetupRequestPayload {
  action?: string
  reason?: string
  request_id: string
  server?: string
}

/** Side agents (`tui_gateway/methods_prompt.py::_spawn_side_agent`). */
export interface SideAgentCompletePayload {
  question?: string
  task_id: string
  text: string
}

export interface PreviewRestartProgressPayload {
  task_id: string
  text: string
}

export interface TerminalOutputPayload {
  chunk?: string
  process_id?: string
}

export interface TerminalClosePayload {
  process_id?: string
}

// ── The map ──────────────────────────────────────────────────────────

/**
 * Backend-emitted notification names. Derived from the `tui_gateway` emitter call
 * sites and pinned to `gateway-events.json`; keep sorted. Adding a name here without
 * the JSON (or vice versa) fails `gateway-events.test.ts`, and a Python emitter that
 * names an event missing from the JSON fails `test_gateway_event_contract.py`.
 */
export const BACKEND_EVENT_NAMES = [
  'agent.terminal.output',
  'approval.request',
  'background.complete',
  'billing.step_up.verification',
  'bot_relay.outbox.pending',
  'browser.controller.cancel',
  'browser.controller.command',
  'browser.progress',
  'btw.complete',
  'clarify.expire',
  'clarify.request',
  'cron.changed',
  'error',
  'gateway.ready',
  'layout.apply',
  'mcp.setup.expire',
  'mcp.setup.request',
  'message.complete',
  'message.delta',
  'message.interim',
  'message.reaction',
  'message.start',
  'moa.aggregating',
  'moa.phase',
  'moa.progress',
  'moa.reference',
  'notice',
  'notification.clear',
  'notification.show',
  'pairing.changed',
  'pane.reveal',
  'pet.changed',
  'pet.generate.progress',
  'pet.hatch.progress',
  'platforms.changed',
  'preview.act.expire',
  'preview.act.request',
  'preview.close',
  'preview.open',
  'preview.read.expire',
  'preview.read.request',
  'preview.restart.complete',
  'preview.restart.progress',
  'reaction',
  'reasoning.available',
  'reasoning.delta',
  'review.summary',
  'secret.expire',
  'secret.request',
  'session.control.update',
  'session.info',
  'session.reclaimed',
  'session.resume_progress',
  'session.title',
  'session.usage',
  'sessions.changed',
  'setup.ready',
  'skin.changed',
  'status.update',
  'subagent.complete',
  'subagent.progress',
  'subagent.spawn_requested',
  'subagent.start',
  'subagent.thinking',
  'subagent.tool',
  'sudo.expire',
  'sudo.request',
  'terminal.close',
  'terminal.read.expire',
  'terminal.read.request',
  'thinking.delta',
  'tip.show',
  'todo.updated',
  'tool.complete',
  'tool.generating',
  'tool.output_risk',
  'tool.start',
  'tour.expire',
  'tour.request',
  'vault.code.expire',
  'vault.code.request',
  'vault.save_login.expire',
  'vault.save_login.request',
  'vault.unlock.expire',
  'vault.unlock.request',
  'voice.interrupted',
  'voice.status',
  'voice.transcript',
  'wake.detected',
  'window.read.expire',
  'window.read.request'
] as const satisfies readonly (keyof BackendGatewayEventMap)[]

export type BackendGatewayEventName = (typeof BACKEND_EVENT_NAMES)[number]

/** Payload per backend-emitted notification `type`. Keys are exactly `BACKEND_EVENT_NAMES`. */
export interface BackendGatewayEventMap {
  'agent.terminal.output': TerminalOutputPayload
  'approval.request': ApprovalRequestPayload
  'background.complete': SideAgentCompletePayload
  'billing.step_up.verification': BillingStepUpVerificationPayload
  'bot_relay.outbox.pending': Record<string, unknown>
  'browser.controller.cancel': Record<string, unknown>
  'browser.controller.command': Record<string, unknown>
  'browser.progress': BrowserProgressPayload
  'btw.complete': SideAgentCompletePayload
  'clarify.expire': RequestExpirePayload
  'clarify.request': ClarifyRequestPayload
  'cron.changed': Record<string, unknown>
  error: ErrorPayload
  'gateway.ready': GatewayReadyPayload
  'layout.apply': Record<string, unknown>
  'mcp.setup.expire': RequestExpirePayload
  'mcp.setup.request': McpSetupRequestPayload
  'message.complete': MessageCompletePayload
  'message.delta': StreamDeltaPayload
  'message.interim': MessageInterimPayload
  'message.reaction': Record<string, unknown>
  'message.start': undefined
  'moa.aggregating': MoaAggregatingPayload
  'moa.phase': MoaPhasePayload
  'moa.progress': MoaProgressPayload
  'moa.reference': MoaReferencePayload
  notice: NoticePayload
  'notification.clear': NotificationClearPayload
  'notification.show': NotificationShowPayload
  'pairing.changed': Record<string, unknown>
  'pane.reveal': Record<string, unknown>
  'pet.changed': Record<string, unknown>
  'pet.generate.progress': Record<string, unknown>
  'pet.hatch.progress': Record<string, unknown>
  'platforms.changed': Record<string, unknown>
  'preview.act.expire': RequestExpirePayload
  'preview.act.request': Record<string, unknown>
  'preview.close': Record<string, unknown>
  'preview.open': Record<string, unknown>
  'preview.read.expire': RequestExpirePayload
  'preview.read.request': Record<string, unknown>
  'preview.restart.complete': SideAgentCompletePayload
  'preview.restart.progress': PreviewRestartProgressPayload
  reaction: ReactionPayload
  'reasoning.available': StreamDeltaPayload
  'reasoning.delta': StreamDeltaPayload
  'review.summary': TextPayload
  'secret.expire': RequestExpirePayload
  'secret.request': SecretRequestPayload
  'session.control.update': SessionControlUpdatePayload
  /** Surface-specific shape (`tui_gateway/server.py::_session_info`); each client narrows. */
  'session.info': Record<string, unknown>
  'session.reclaimed': SessionReclaimedPayload
  'session.resume_progress': SessionResumeProgressPayload
  'session.title': SessionTitlePayload
  'session.usage': SessionUsagePayload
  'sessions.changed': Record<string, unknown>
  'setup.ready': Record<string, unknown>
  'skin.changed': HermesSkin
  'status.update': StatusUpdatePayload
  'subagent.complete': SubagentEventPayload
  'subagent.progress': SubagentEventPayload
  'subagent.spawn_requested': SubagentEventPayload
  'subagent.start': SubagentEventPayload
  'subagent.thinking': SubagentEventPayload
  'subagent.tool': SubagentEventPayload
  'sudo.expire': RequestExpirePayload
  'sudo.request': SudoRequestPayload
  'terminal.close': TerminalClosePayload
  'terminal.read.expire': RequestExpirePayload
  'terminal.read.request': Record<string, unknown>
  'thinking.delta': StreamDeltaPayload
  'tip.show': Record<string, unknown>
  'todo.updated': TodoStatePayload
  'tool.complete': ToolCompletePayload
  'tool.generating': ToolGeneratingPayload
  'tool.output_risk': ToolOutputRiskPayload
  'tool.start': ToolStartPayload
  'tour.expire': RequestExpirePayload
  'tour.request': Record<string, unknown>
  'vault.code.expire': RequestExpirePayload
  'vault.code.request': VaultCodeRequestPayload
  'vault.save_login.expire': RequestExpirePayload
  'vault.save_login.request': Record<string, unknown>
  'vault.unlock.expire': RequestExpirePayload
  'vault.unlock.request': VaultUnlockRequestPayload
  'voice.interrupted': Record<string, unknown>
  'voice.status': VoiceStatusPayload
  'voice.transcript': VoiceTranscriptPayload
  'wake.detected': WakeDetectedPayload
  'window.read.expire': RequestExpirePayload
  'window.read.request': Record<string, unknown>
}

/**
 * Client-local synthetic events. Never emitted by `tui_gateway`; the Ink TUI's
 * `gatewayClient` publishes them into the same handler stream to report transport
 * state. Excluded from `gateway-events.json` on purpose.
 */
export interface ClientLocalGatewayEventMap {
  'dashboard.new_session_requested': { reason?: string }
  'gateway.protocol_error': { preview?: string }
  'gateway.reconnecting': { attempt?: number; delay_ms?: number }
  'gateway.start_timeout': { cwd?: string; python?: string; stderr_tail?: string }
  'gateway.stderr': { line: string }
}

export interface GatewayEventMap extends BackendGatewayEventMap, ClientLocalGatewayEventMap {}

export type GatewayEventName = keyof GatewayEventMap

/** One `event` notification's `params`. */
export interface GatewayEvent<K extends GatewayEventName = GatewayEventName> {
  /** Registry connection whose socket delivered the event (renderer-side tag;
   * absent for the local/legacy primary path). */
  connectionId?: string
  payload?: GatewayEventMap[K]
  /** Renderer-side source tag added by the Desktop gateway registry. */
  profile?: string
  /** Per-session monotonic counter stamped by `tui_gateway/event_replay.py::_stamp_event`;
   *  absent on session-less broadcasts. */
  seq?: number
  session_id?: string
  type: K
}

// ── RPC responses shared across surfaces ─────────────────────────────

/** `hermes_cli/inventory.py` one `model.options` provider row (union of every field the
 *  backend sets; `pricing_pending` / `free_tier_pending` mark the cached-only fail-closed path). */
export interface ModelOptionProvider {
  /** User-defined providers only: every accepted identity for this endpoint
   *  (bare config key, `custom:<key>`, normalized display name, …). A session's
   *  `model.options` reports the canonical `custom:<key>` form, so "is this row
   *  the current provider?" must check membership here, not slug equality. */
  aliases?: string[]
  /** OpenAI-compatible endpoint for a user-defined provider. The backend
   *  exposes this as `api_url`; model assignments send it back as `base_url`. */
  api_url?: string
  /** Auth flow for an unconfigured provider: "api_key" can be activated inline
   *  by pasting `key_env`; anything else (oauth_*, external, aws_sdk, …) needs
   *  the `hermes model` CLI / onboarding OAuth flow. */
  auth_type?: string
  /** True when the provider has usable credentials. False for canonical
   *  providers surfaced by `include_unconfigured` that the user hasn't set up
   *  yet — render these with a setup affordance instead of hiding them. */
  authenticated?: boolean
  /** Per-model option support, keyed by model id (present when the picker
   *  requested capabilities). Lets the UI gate fast/reasoning controls. */
  capabilities?: Record<string, ModelCapabilities>
  /** Curated shortlist (one flagship per lab) the picker shows by default for
   *  aggregator providers that serve dozens of models across many labs. */
  featured_models?: string[]
  /** Nous only: whether the current account is on the free plan. */
  free_tier?: boolean
  /** Nous only, cached-only inventory: entitlement unknown, every model rendered locked. */
  free_tier_pending?: boolean
  /** True for the free-tier route's own provider row (no account behind it).
   *  Never match this row by `name` — the label is copy and can change. */
  free_tier_row?: boolean
  is_current?: boolean
  /** True for providers defined via the user's `providers:` config block. */
  is_user_defined?: boolean
  /** Env var to paste an API key into, for unconfigured `api_key` providers. */
  key_env?: string
  models?: string[]
  name: string
  /** Per-model pricing keyed by model id (present when the picker requested
   *  pricing and the provider supports live pricing). */
  pricing?: Record<string, ModelPricing>
  /** Cached-only inventory: pricing not fetched yet. */
  pricing_pending?: boolean
  slug: string
  source?: string
  total_models?: number
  /** Nous only: paid models a free-tier user cannot select (shown disabled). */
  unavailable_models?: string[]
  warning?: string
}

export interface ModelPricing {
  /** Formatted $/Mtok cached-input price, or null when the model has none. */
  cache: null | string
  /** Sale: rounded percent off list when gateway sends pricing.original. */
  discount_percent?: number
  /** True when the model costs nothing (free tier eligible). */
  free: boolean
  /** Formatted $/Mtok input price, e.g. "$3.00", or "free", or "" if unknown. */
  input: string
  /** Formatted $/Mtok output price. */
  output: string
  /** Sale: formatted pre-discount input $/Mtok ("was"). */
  was_input?: string
  /** Sale: formatted pre-discount output $/Mtok ("was"). */
  was_output?: string
}

export interface ModelCapabilities {
  /** False when the route rejects a reasoning disable ("mandatory" in the
   *  provider catalog), so the Thinking toggle must not be offered. */
  can_disable_reasoning?: boolean
  fast: boolean
  reasoning: boolean
}

export interface ModelOptionsResponse {
  model?: string
  provider?: string
  providers?: ModelOptionProvider[]
}

/** `tui_gateway/methods_session.py::_session_row_summary` — one `session.list` row. */
export interface SessionListItem {
  id: string
  message_count: number
  preview: string
  /** The runtime id this stored session is currently attached to, when live. */
  resolved_id?: string
  source?: string
  started_at: number
  title: string
}

export interface SessionListResponse {
  sessions?: SessionListItem[]
}

/** Transcript row as projected by the gateway (`session.resume` / `session.activate`). */
export interface GatewayTranscriptMessage {
  args?: unknown
  context?: string
  display_kind?: string
  display_metadata?: unknown
  name?: string
  role: 'assistant' | 'system' | 'tool' | 'user'
  text?: string
}

export interface SessionInflightTurn {
  assistant?: string
  correction_offsets?: number[]
  corrections?: string[]
  error?: string
  error_surface?: ErrorSurface
  recoverable?: boolean
  status?: string
  streaming?: boolean
  user?: string
}

/** `tui_gateway/methods_session.py::_resume_response`. `info` is surface-specific
 *  (`SessionInfo` in the TUI, `SessionRuntimeInfo` on Desktop); narrow at the call site. */
export interface SessionResumeResponse<Info = Record<string, unknown>, Message = GatewayTranscriptMessage> {
  /** Present when the backend found a fresh crash-interrupted turn and scheduled its
   *  automatic continuation; the turn arrives as a normal message.start stream. */
  auto_continue?: { attempt: number; interrupted_at: number }
  /** Deferred hydration: history arrives via `session.resume_progress`. */
  hydrating?: boolean
  inflight?: null | SessionInflightTurn
  info?: Info
  message_count?: number
  messages: Message[]
  /** `omit_messages` resume: the client still learns the stored size. */
  messages_omitted?: boolean
  resumed?: string
  running?: boolean
  session_id: string
  session_key?: string
  started_at?: number
  status?: string
  todo_state?: TodoStatePayload
}
