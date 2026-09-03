import type { ThreadMessageLike } from '@assistant-ui/react'
import { type BillingBlock } from '@hermes/shared'

import type { ErrorSurface } from '@/lib/error-surface'
import type { MessageReaction, SessionMessage, UsageStats } from '@/types/hermes'

export interface TimelinePartMetadata {
  /** Unix seconds when this visible activity segment began. Fractional values
   * preserve the millisecond precision available on live gateway events. */
  timestamp?: number
  /** Unix seconds when this segment stopped or handed off to the next one. */
  completedAt?: number
}

export type ChatMessagePart = Exclude<ThreadMessageLike['content'], string>[number] & TimelinePartMetadata

export type ChatMessage = {
  id: string
  role: SessionMessage['role']
  parts: ChatMessagePart[]
  timestamp?: number
  completedAt?: number
  pending?: boolean
  error?: string
  /** Structured layer descriptor for a failed turn (parsed error_surface).
   *  Drives the error card's layer label + actions; absent on older
   *  backends, where the card falls back to generic copy. */
  errorSurface?: ErrorSurface
  branchGroupId?: string
  hidden?: boolean
  /** Sealed mid-turn commentary (`message.interim`) — rendered without the
   *  action footer so only the turn's final reply carries copy/refresh, and
   *  the live view matches rehydration (which merges the turn into one bubble). */
  interim?: boolean
  /** Whole-turn wall-clock seconds (message.start → message.complete),
   *  stamped by the desktop when it watched the turn run. Absent for
   *  messages hydrated from history — the backend doesn't persist it. */
  durationS?: number
  /** Composer attachment ref strings (`@file:...`, `@image:...`) sent with this user message. */
  attachmentRefs?: string[]
  /** Durable backend `messages.id`. Absent until the row is persisted. */
  rowId?: number
  /** Emoji reactions on this message — one per author (see MessageReaction). */
  reactions?: MessageReaction[]
}

export type GatewayEventPayload = {
  /** Unix seconds supplied by tests/newer gateways; the desktop falls back to
   * its local receipt clock when older gateways omit it. */
  timestamp?: number
  text?: string
  rendered?: string
  status?: string
  message?: string
  id?: string
  name?: string
  tool_id?: string
  tool_call_id?: string
  args?: unknown
  arguments?: unknown
  context?: string
  input?: unknown
  preview?: string
  result?: unknown
  summary?: string
  error?: string | boolean
  // message.complete with status "error" — structured {layer, code, retryable}
  // descriptor naming which stack layer failed (agent/error_surface.py).
  // Absent on older gateways; consumers must fall back to string heuristics.
  error_surface?: unknown
  inline_diff?: string
  duration_s?: number
  todos?: unknown
  revision?: number
  model?: string
  provider?: string
  reasoning_effort?: string
  service_tier?: string
  fast?: boolean
  approval_mode?: string
  yolo?: boolean
  running?: boolean
  turn_started_at?: number | null
  cwd?: string
  branch?: string
  terminal_backend?: string
  credential_warning?: string
  install_warning?: string
  personality?: string
  usage?: Partial<UsageStats>
  // agent.terminal.output — live chunk for a read-only agent terminal tab
  process_id?: string
  chunk?: string
  // clarify.request
  request_id?: string
  question?: string
  // btw.complete / background.complete — id of the side/background task
  task_id?: string
  choices?: string[] | null
  multi_select?: boolean
  // clarify.request batch form: questions replaces question/choices, and
  // answers (qid → locked answer) rides along on reconnect replay only.
  questions?: unknown
  answers?: Record<string, unknown>
  // mcp.setup.request (setup_mcp tool — inline MCP consent card)
  server?: string
  action?: string
  reason?: string
  // approval.request (dangerous command / execute_code) — session-keyed
  command?: string
  description?: string
  // False when a tirith content-security warning forbids a permanent allow.
  allow_permanent?: boolean
  smart_denied?: boolean
  // secret.request (skill credential capture)
  env_var?: string
  prompt?: string
  // terminal.read.request / preview.read.request (GUI agent reading the
  // in-app terminal pane or the browser/preview pane)
  start?: number
  count?: number
  // status.update (kind=process → background process completion/watch-match)
  kind?: string
  // pane.reveal (agent focusing a desktop pane via the focus_pane tool)
  pane?: string
  // layout.apply (agent applying a layout preset via the apply_layout tool)
  preset?: string
  // tour.request (tour tool — agent-guided driver.js walkthrough). `action`
  // and `steps` name the tour verb and step list; `surface` picks the app's
  // own DOM vs the preview pane's guest page. tip.show (tip tool — one accent
  // bubble with an arrow, no overlay) adds no fields of its own: it reuses
  // `selector`/`side` here plus `text`/`title`, and carries no request_id
  // because a tip is fire-and-forget.
  surface?: string
  selector?: string
  side?: string
  steps?: unknown
  step_index?: number
  // preview.act.request (drive_preview tool — agent clicking/typing/scrolling in
  // the in-app browser). `action` names the verb and `selector` is shared with
  // tour above; `ref` addresses an element from the last inventory.
  ref?: string
  submit?: boolean
  key?: string
  amount?: number
  to?: string
  max?: number
  // message.reaction (agent reacting via the react_to_message tool) — the
  // durable messages.id, that row's full reaction list after the write, and
  // the row's role so a live (not-yet-round-tripped) message can be matched.
  row_id?: number
  reactions?: MessageReaction[]
  role?: string
  // session.title (live auto-title push) — stored session id + generated title
  session_id?: string
  title?: string
  // session.info — the stored (durable) session id for this runtime session.
  // Lets the desktop app map runtime→stored for background sessions it hasn't
  // opened, so the sidebar working indicator updates without opening the chat.
  stored_session_id?: string
  // moa.reference / moa.aggregating (Mixture of Agents per-model relay)
  label?: string
  index?: number
  aggregator?: string
  // moa.progress / moa.phase (Mixture of Agents fan-out progress relay)
  refs_done?: number
  refs_total?: number
  phase?: string
  // message.complete — signals the final text was already previewed via
  // interim_assistant_callback, so the UI can settle instead of duplicating.
  response_previewed?: boolean
  // message.complete — a transform_llm_output hook rewrote the final text
  // after streaming finished. The text authoritatively replaces the current
  // turn's streamed assistant text even without a prefix relationship.
  response_transformed?: boolean
  // message.complete with status "error" — `text` is streamed partial output
  // (keep it visible), not the error string.
  partial?: boolean
  // message.complete with status "error" — the failed turn was retained
  // backend-side and will replay through session.resume's inflight payload.
  recoverable?: boolean
  // Structured billing wall forwarded on message.complete when a turn fails
  // with FailoverReason.billing (shape mirrors @hermes/shared BillingBlock).
  billing?: BillingBlock
  failure_reason?: string
}
