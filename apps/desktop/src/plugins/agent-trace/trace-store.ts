/**
 * AgentTrace store — a pure event reducer over the gateway's live frames plus
 * snapshot hydration doors. The reducer (`applyEvent` and the merge helpers)
 * is side-effect-free so tests drive it with synthetic frames; `bindTrace`
 * wires it to `ctx.onEvent`, and the fetch helpers feed React Query caches
 * that merge their results back through the same pure helpers.
 *
 * Memory is bounded: sessions, tool rows and child progress rows are capped,
 * long text is truncated with a flag, and only the focused/active surface
 * keeps growing. Nothing here polls — events drive updates; the mounted page
 * and pane refresh snapshots on a slow React Query interval.
 */

import { atom, host, type PluginContext } from '@hermes/plugin-sdk'

import type {
  SelectedStep,
  SessionTrace,
  SubagentNode,
  SubagentNodeStatus,
  ToolCall,
  TraceEvent,
  TraceState,
  WireActiveSubagent,
  WireDelegationStatus,
  WireHistoryRow,
  WireSessionUsage,
  WireSubagentEvent,
  WireSubagentList,
  WireSubagentSnapshot,
  WireToolComplete,
  WireToolProgress,
  WireToolStart,
  WireUsage
} from './types'
import { isTerminalSubagent } from './types'

// ── Caps (memory bounds) ────────────────────────────────────────────────────

const MAX_SESSIONS = 24
const MAX_TOOLS_PER_SESSION = 400
const MAX_SUBAGENTS_PER_SESSION = 60
const MAX_CHILD_TOOL_ROWS = 120
const TEXT_CAP = 200_000
const ARGS_CAP = 100_000
const THINKING_CAP = 2_000

// ── Atoms ───────────────────────────────────────────────────────────────────

export const $trace = atom<TraceState>({ sessions: [] })
export const $selectedStep = atom<SelectedStep | null>(null)
/** 'focused' follows the interacted tile; 'all' shows every traced session. */
export const $traceScope = atom<'all' | 'focused'>('focused')
export const $traceFilter = atom('')
/** Frozen view: live frames are dropped while paused (inspection mode). */
export const $tracePaused = atom(false)

export const EMPTY_TRACE: TraceState = { sessions: [] }

// ── Small helpers ───────────────────────────────────────────────────────────

function capText(text: string, cap: number): { text: string; truncated: boolean } {
  return text.length > cap
    ? { text: `${text.slice(0, cap)}\n… [truncated ${text.length - cap} chars]`, truncated: true }
    : { text, truncated: false }
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

/** Mirror of the backend's failure classification (`tool_progress.
 *  _tool_result_needs_user`): success/ok false, a non-empty error string, or
 *  a non-zero numeric exit_code. */
export function detectToolFailure(
  result: unknown,
  resultText?: string | null
): { failed: boolean; error: string | null } {
  let data = asRecord(result)

  if (!data && typeof resultText === 'string' && resultText) {
    try {
      data = asRecord(JSON.parse(resultText))
    } catch {
      data = null
    }
  }

  if (!data) {
    return { failed: false, error: null }
  }

  const error = typeof data.error === 'string' && data.error.trim() ? data.error : null
  const exitCode = typeof data.exit_code === 'number' && !Number.isNaN(data.exit_code) ? data.exit_code : null
  const failed = data.success === false || data.ok === false || error !== null || (exitCode !== null && exitCode !== 0)

  if (!failed) {
    return { failed: false, error: null }
  }

  return {
    failed: true,
    error: error ?? (exitCode !== null && exitCode !== 0 ? `Command exited with code ${exitCode}` : null)
  }
}

function resultToText(result: unknown, resultText?: string | null): string | null {
  if (typeof resultText === 'string' && resultText) {
    return resultText
  }

  if (typeof result === 'string') {
    return result
  }

  if (result === undefined || result === null) {
    return null
  }

  try {
    return JSON.stringify(result)
  } catch {
    return String(result)
  }
}

function rawFrame(payload: unknown): string | undefined {
  try {
    const text = JSON.stringify(payload)

    return text === undefined ? undefined : capText(text, TEXT_CAP).text
  } catch {
    return undefined
  }
}

function mergeUsage(existing: WireUsage | null | undefined, incoming: WireUsage | null | undefined): WireUsage | null {
  if (!incoming) {
    return existing ?? null
  }

  return { ...(existing ?? {}), ...incoming }
}

/** Fold the deprecated top-level `session.usage` counters into `Usage`. */
function usageFromFrame(payload: WireSessionUsage): WireUsage | null {
  if (payload.usage) {
    return payload.usage
  }

  if (payload.context_tokens == null && payload.input_tokens == null && payload.total_tokens == null) {
    return null
  }

  const folded: WireUsage = {}

  if (payload.context_tokens != null) {
    folded.context_used = payload.context_tokens
  }

  if (payload.context_length != null) {
    folded.context_max = payload.context_length
  }

  if (payload.input_tokens != null) {
    folded.input = payload.input_tokens
  }

  if (payload.output_tokens != null) {
    folded.output = payload.output_tokens
  }

  if (payload.total_tokens != null) {
    folded.total = payload.total_tokens
  }

  if (payload.api_calls != null) {
    folded.calls = payload.api_calls
  }

  if (payload.avg_latency_s != null) {
    folded.avg_latency_s = payload.avg_latency_s
  }

  if (payload.active_subagents != null) {
    folded.active_subagents = payload.active_subagents
  }

  if (payload.model != null) {
    folded.model = payload.model
  }

  return folded
}

// ── Session / node upserts (all return NEW objects; inputs never mutated) ───

function upsertSession(
  state: TraceState,
  sessionId: string,
  at: number,
  meta: { profile?: string | null; connectionId?: string | null }
): { state: TraceState; session: SessionTrace } {
  const index = state.sessions.findIndex(s => s.sessionId === sessionId)
  const existing = index >= 0 ? state.sessions[index] : null

  const session: SessionTrace = existing
    ? { ...existing, tools: [...existing.tools], subagents: [...existing.subagents], lastActivityAt: at }
    : {
        sessionId,
        firstSeenAt: at,
        lastActivityAt: at,
        turnActive: false,
        tools: [],
        subagents: []
      }

  if (meta.profile) {
    session.profile = meta.profile
  }

  if (meta.connectionId) {
    session.connectionId = meta.connectionId
  }

  let sessions = index >= 0 ? state.sessions.map((s, i) => (i === index ? session : s)) : [...state.sessions, session]

  if (sessions.length > MAX_SESSIONS) {
    // Drop the least-recently-active sessions, never the one just touched.
    const victims = sessions
      .filter(s => s.sessionId !== sessionId)
      .sort((a, b) => a.lastActivityAt - b.lastActivityAt)
      .slice(0, sessions.length - MAX_SESSIONS)
      .map(s => s.sessionId)

    sessions = sessions.filter(s => !victims.includes(s.sessionId))
  }

  return { state: { sessions }, session }
}

function replaceSession(state: TraceState, session: SessionTrace): TraceState {
  return { sessions: state.sessions.map(s => (s.sessionId === session.sessionId ? session : s)) }
}

function upsertTool(session: SessionTrace, tool: ToolCall): void {
  const index = session.tools.findIndex(t => t.id === tool.id)

  if (index >= 0) {
    session.tools[index] = { ...session.tools[index], ...tool }
  } else {
    session.tools.push(tool)
  }

  if (session.tools.length > MAX_TOOLS_PER_SESSION) {
    // Evict oldest SETTLED calls first; running calls stay visible.
    const settled = session.tools.filter(t => t.status === 'done' || t.status === 'error')
    const victim = settled.sort((a, b) => (a.endedAt ?? a.startedAt ?? 0) - (b.endedAt ?? b.startedAt ?? 0))[0]
    session.tools = session.tools.filter(t => t !== victim)
  }
}

function upsertSubagent(session: SessionTrace, node: SubagentNode): void {
  const index = session.subagents.findIndex(s => s.id === node.id)

  if (index >= 0) {
    const existing = session.subagents[index]

    // A terminal record is never reopened by a stale snapshot.
    const merged =
      isTerminalSubagent(existing.status) && !isTerminalSubagent(node.status)
        ? { ...node, ...existing, tools: existing.tools }
        : { ...existing, ...node }

    session.subagents[index] = merged
  } else {
    session.subagents.push(node)
  }

  if (session.subagents.length > MAX_SUBAGENTS_PER_SESSION) {
    const settled = session.subagents.filter(s => isTerminalSubagent(s.status))
    const victim = settled.sort((a, b) => (a.endedAt ?? a.startedAt ?? 0) - (b.endedAt ?? b.startedAt ?? 0))[0]

    if (victim) {
      session.subagents = session.subagents.filter(s => s !== victim)
    }
  }
}

function subagentIdFrom(payload: WireSubagentEvent, sessionId: string): string {
  return payload.subagent_id || `${sessionId}:${payload.delegation_id ?? 'd'}:${payload.task_index}`
}

function nodeFromSubagentEvent(
  session: SessionTrace,
  payload: WireSubagentEvent,
  id: string,
  at: number
): SubagentNode {
  const existing = session.subagents.find(s => s.id === id)

  return (
    existing ?? {
      id,
      depth: payload.depth ?? 0,
      goal: payload.goal,
      taskIndex: payload.task_index,
      taskCount: payload.task_count,
      status: 'spawn_requested',
      toolCount: 0,
      inputTokens: 0,
      outputTokens: 0,
      reasoningTokens: 0,
      apiCalls: 0,
      outputTail: [],
      tools: [],
      startedAt: at
    }
  )
}

function applySubagentRollup(node: SubagentNode, payload: WireSubagentEvent): void {
  if (payload.tool_count != null) {
    node.toolCount = payload.tool_count
  }

  if (payload.input_tokens != null) {
    node.inputTokens = payload.input_tokens
  }

  if (payload.output_tokens != null) {
    node.outputTokens = payload.output_tokens
  }

  if (payload.reasoning_tokens != null) {
    node.reasoningTokens = payload.reasoning_tokens
  }

  if (payload.api_calls != null) {
    node.apiCalls = payload.api_calls
  }

  if (payload.model) {
    node.model = payload.model
  }

  if (payload.parent_id !== undefined) {
    node.parentId = payload.parent_id
  }

  if (payload.depth != null) {
    node.depth = payload.depth
  }

  if (payload.child_session_id) {
    node.childSessionId = payload.child_session_id
  }

  if (payload.delegation_id) {
    node.delegationId = payload.delegation_id
  }
}

// ── The reducer ─────────────────────────────────────────────────────────────
// One small handler per frame family keeps every function under the repo's
// code-health caps; `applyEvent` is a pure dispatch over them.

interface FrameContext {
  sessionId: string
  at: number
  type: string
  meta: { profile?: string | null; connectionId?: string | null }
}

type FrameHandler = (state: TraceState, ctx: FrameContext, payload: unknown) => TraceState

function onToolGenerating(state: TraceState, ctx: FrameContext, payload: unknown): TraceState {
  const p = payload as { name?: string } | undefined

  if (!p?.name) {
    return state
  }

  const up = upsertSession(state, ctx.sessionId, ctx.at, ctx.meta)
  up.session.generating = { name: p.name, startedAt: ctx.at }

  return replaceSession(up.state, up.session)
}

/** Cap oversized argument payloads: keep a truncated text copy, drop the object. */
function capArgs(
  args: Record<string, unknown> | null,
  argsText: string | null
): { args: Record<string, unknown> | null; argsText: string | null; truncated: boolean } {
  if (!args) {
    return { args: null, argsText, truncated: false }
  }

  const serialized = JSON.stringify(args)

  if (serialized.length <= ARGS_CAP) {
    return { args, argsText, truncated: false }
  }

  return { args: null, argsText: capText(serialized, ARGS_CAP).text, truncated: true }
}

function onToolStart(state: TraceState, ctx: FrameContext, payload: unknown): TraceState {
  const p = payload as WireToolStart | undefined

  if (!p?.tool_id || !p.name) {
    return state
  }

  const up = upsertSession(state, ctx.sessionId, ctx.at, ctx.meta)
  up.session.generating = null
  const capped = capArgs(p.args ?? null, p.args_text ?? null)
  upsertTool(up.session, {
    id: p.tool_id,
    name: p.name,
    status: 'running',
    origin: 'live',
    args: capped.args,
    argsText: capped.argsText,
    preview: p.preview ?? p.context ?? null,
    labels: p.labels ?? null,
    startedAt: ctx.at,
    truncated: capped.truncated
  })

  return replaceSession(up.state, up.session)
}

function onToolProgress(state: TraceState, ctx: FrameContext, payload: unknown): TraceState {
  const p = payload as WireToolProgress | undefined

  if (!p?.tool_id) {
    return state
  }

  const up = upsertSession(state, ctx.sessionId, ctx.at, ctx.meta)
  const existing = up.session.tools.find(t => t.id === p.tool_id)

  if (!existing) {
    return state
  }

  upsertTool(up.session, {
    ...existing,
    progress: { value: p.progress, total: p.total ?? null, message: p.message ?? null }
  })

  return replaceSession(up.state, up.session)
}

/** The error trace a failed call surfaces in the drawer. */
function failureTrace(
  failure: { failed: boolean; error: string | null },
  summary?: string | null,
  text?: string | null
): string | null {
  if (!failure.failed) {
    return null
  }

  return failure.error ?? summary ?? text?.slice(0, 200) ?? 'Tool call failed'
}

/** Fold a `tool.complete` frame (plus any live start row) into a settled call. */
function buildCompletedTool(p: WireToolComplete, existing: ToolCall | undefined, at: number): ToolCall {
  const fullText = resultToText(p.result, p.result_text)
  const capped = fullText ? capText(fullText, TEXT_CAP) : null
  const failure = detectToolFailure(p.result, fullText)
  const startedAt = existing?.startedAt ?? null

  return {
    id: p.tool_id,
    name: p.name || existing?.name || 'tool',
    status: failure.failed ? 'error' : 'done',
    origin: 'live',
    args: p.args ?? existing?.args ?? null,
    argsText: existing?.argsText ?? null,
    preview: existing?.preview ?? null,
    labels: p.labels ?? existing?.labels ?? null,
    startedAt,
    endedAt: at,
    durationS: p.duration_s ?? (startedAt != null ? Math.max(0, at - startedAt) : null),
    resultText: capped?.text ?? null,
    summary: p.summary ?? null,
    error: failureTrace(failure, p.summary, capped?.text),
    resultChars: fullText?.length ?? null,
    truncated: capped?.truncated ?? existing?.truncated ?? false,
    progress: null,
    raw: rawFrame(p)
  }
}

function onToolComplete(state: TraceState, ctx: FrameContext, payload: unknown): TraceState {
  const p = payload as WireToolComplete | undefined

  if (!p?.tool_id) {
    return state
  }

  const up = upsertSession(state, ctx.sessionId, ctx.at, ctx.meta)
  const existing = up.session.tools.find(t => t.id === p.tool_id)
  upsertTool(up.session, buildCompletedTool(p, existing, ctx.at))

  return replaceSession(up.state, up.session)
}

/** Terminal fold of a `subagent.complete` frame. */
function finishSubagent(node: SubagentNode, p: WireSubagentEvent, at: number): void {
  node.status = p.status ?? 'completed'
  node.endedAt = at
  node.durationS = p.duration_seconds ?? (node.startedAt != null ? Math.max(0, at - node.startedAt) : null)
  node.summary = p.summary ?? node.summary ?? null

  if (p.output_tail) {
    node.outputTail = p.output_tail
  }

  if (isTerminalSubagent(node.status) && node.status !== 'completed') {
    node.error = p.summary ?? p.text ?? null
  }
}

function onSubagentFrame(state: TraceState, ctx: FrameContext, payload: unknown): TraceState {
  const p = payload as WireSubagentEvent | undefined

  if (!p) {
    return state
  }

  const up = upsertSession(state, ctx.sessionId, ctx.at, ctx.meta)
  const id = subagentIdFrom(p, ctx.sessionId)

  const node = {
    ...nodeFromSubagentEvent(up.session, p, id, ctx.at),
    tools: [...(up.session.subagents.find(s => s.id === id)?.tools ?? [])]
  }

  node.goal = p.goal || node.goal
  node.taskIndex = p.task_index
  node.taskCount = p.task_count
  applySubagentRollup(node, p)
  const type = ctx.type

  if (type === 'subagent.spawn_requested') {
    if (!isTerminalSubagent(node.status)) {
      node.status = 'queued'
    }

    node.startedAt = node.startedAt ?? ctx.at
  } else if (type === 'subagent.start') {
    node.status = 'running'
    node.startedAt = ctx.at
  } else if (type === 'subagent.tool') {
    if (p.tool_name) {
      node.lastTool = p.tool_name
      node.tools.push({
        id: `${id}:t${node.tools.length}`,
        name: p.tool_name,
        status: 'done',
        origin: 'live',
        preview: p.tool_preview ?? null,
        subagentId: id,
        endedAt: ctx.at
      })

      if (node.tools.length > MAX_CHILD_TOOL_ROWS) {
        node.tools = node.tools.slice(node.tools.length - MAX_CHILD_TOOL_ROWS)
      }
    }
  } else if (type === 'subagent.thinking') {
    if (p.text) {
      node.thinkingTail = `${node.thinkingTail ?? ''}${p.text}`.slice(-THINKING_CAP)
    }
  } else if (type === 'subagent.complete') {
    finishSubagent(node, p, ctx.at)
  }

  upsertSubagent(up.session, node)

  return replaceSession(up.state, up.session)
}

function onMessageStart(state: TraceState, ctx: FrameContext): TraceState {
  const up = upsertSession(state, ctx.sessionId, ctx.at, ctx.meta)
  up.session.turnActive = true

  return replaceSession(up.state, up.session)
}

function onMessageComplete(state: TraceState, ctx: FrameContext, payload: unknown): TraceState {
  const p = payload as { usage?: WireUsage | null; error?: string | null } | undefined
  const up = upsertSession(state, ctx.sessionId, ctx.at, ctx.meta)
  up.session.turnActive = false
  up.session.generating = null
  up.session.usage = mergeUsage(up.session.usage, p?.usage)

  if (p?.error) {
    up.session.lastError = { message: p.error, at: ctx.at }
  }

  return replaceSession(up.state, up.session)
}

function onSessionUsage(state: TraceState, ctx: FrameContext, payload: unknown): TraceState {
  const p = payload as WireSessionUsage | undefined
  const usage = p ? usageFromFrame(p) : null

  if (!usage) {
    return state
  }

  const up = upsertSession(state, ctx.sessionId, ctx.at, ctx.meta)
  up.session.usage = mergeUsage(up.session.usage, usage)

  if (p?.model && !up.session.model) {
    up.session.model = p.model
  }

  if (p?.provider && !up.session.provider) {
    up.session.provider = p.provider
  }

  return replaceSession(up.state, up.session)
}

function onSessionInfo(state: TraceState, ctx: FrameContext, payload: unknown): TraceState {
  const p = payload as
    { model?: string; provider?: string; title?: string; running?: boolean; usage?: WireUsage | null } | undefined

  if (!p) {
    return state
  }

  const up = upsertSession(state, ctx.sessionId, ctx.at, ctx.meta)

  if (p.model) {
    up.session.model = p.model
  }

  if (p.provider) {
    up.session.provider = p.provider
  }

  if (p.title) {
    up.session.title = p.title
  }

  if (typeof p.running === 'boolean') {
    up.session.turnActive = p.running
  }

  up.session.usage = mergeUsage(up.session.usage, p.usage)

  return replaceSession(up.state, up.session)
}

function onSessionTitle(state: TraceState, ctx: FrameContext, payload: unknown): TraceState {
  const p = payload as { title?: string } | undefined

  if (!p?.title) {
    return state
  }

  const up = upsertSession(state, ctx.sessionId, ctx.at, ctx.meta)
  up.session.title = p.title

  return replaceSession(up.state, up.session)
}

function onErrorFrame(state: TraceState, ctx: FrameContext, payload: unknown): TraceState {
  const p = payload as { message?: string } | undefined

  if (!p?.message) {
    return state
  }

  const up = upsertSession(state, ctx.sessionId, ctx.at, ctx.meta)
  up.session.lastError = { message: p.message, at: ctx.at }

  return replaceSession(up.state, up.session)
}

const FRAME_HANDLERS: Record<string, FrameHandler> = {
  'tool.generating': onToolGenerating,
  'tool.start': onToolStart,
  'tool.progress': onToolProgress,
  'tool.complete': onToolComplete,
  'subagent.spawn_requested': onSubagentFrame,
  'subagent.start': onSubagentFrame,
  'subagent.progress': onSubagentFrame,
  'subagent.tool': onSubagentFrame,
  'subagent.thinking': onSubagentFrame,
  'subagent.complete': onSubagentFrame,
  'message.start': onMessageStart,
  'message.complete': onMessageComplete,
  'session.usage': onSessionUsage,
  'session.info': onSessionInfo,
  'session.title': onSessionTitle,
  error: onErrorFrame
}

/** Apply one normalized gateway frame. Pure: returns a new state (or the
 *  same reference when the frame changes nothing). */
export function applyEvent(state: TraceState, event: TraceEvent): TraceState {
  const sessionId = event.sessionId
  const handler = FRAME_HANDLERS[event.type]

  if (!handler || !sessionId) {
    return state
  }

  return handler(
    state,
    { sessionId, at: event.at, type: event.type, meta: { profile: event.profile, connectionId: event.connectionId } },
    event.payload
  )
}

// ── Snapshot merges (RPC results → the same model) ──────────────────────────

function nodeFromActive(record: WireActiveSubagent, at: number): SubagentNode {
  return {
    id: record.subagent_id,
    parentId: record.parent_id ?? null,
    depth: record.depth ?? 0,
    goal: record.goal ?? '',
    taskIndex: 0,
    taskCount: 1,
    model: record.model ?? null,
    status: (record.status as SubagentNodeStatus) || 'running',
    startedAt: record.started_at ?? at,
    toolCount: record.tool_count ?? 0,
    delegationId: record.delegation_id ?? null,
    inputTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    apiCalls: 0,
    outputTail: [],
    tools: []
  }
}

/** Merge `delegation.status` — the profile-wide running tree. */
export function mergeDelegationStatus(state: TraceState, status: WireDelegationStatus, at: number): TraceState {
  let next = state

  for (const record of status.active ?? []) {
    const owner = record.owner_agent_session_id

    if (!owner || !record.subagent_id) {
      continue
    }

    const up = upsertSession(next, owner, at, {})
    upsertSubagent(up.session, nodeFromActive(record, at))
    next = replaceSession(up.state, up.session)
  }

  return next
}

/** First non-nullish value — keeps snapshot merges free of long ?? chains. */
function pick3<T>(a: T | null | undefined, b: T | null | undefined, fallback: T): T {
  if (a !== null && a !== undefined) {
    return a
  }

  if (b !== null && b !== undefined) {
    return b
  }

  return fallback
}

/** Fold one `subagent.list` snapshot over any node already traced live. */
function nodeFromSnapshot(snap: WireSubagentSnapshot, existing: SubagentNode | undefined, at: number): SubagentNode {
  return {
    ...(existing ?? {
      id: snap.subagent_id,
      taskIndex: 0,
      taskCount: 1,
      inputTokens: 0,
      outputTokens: 0,
      reasoningTokens: 0,
      apiCalls: 0,
      outputTail: [],
      tools: []
    }),
    parentId: pick3(snap.parent_id, existing?.parentId, null),
    depth: pick3(snap.depth, existing?.depth, 0),
    goal: snap.goal || existing?.goal || '',
    model: pick3(snap.model, existing?.model, null),
    status: pick3(snap.status, existing?.status, 'running' as SubagentNodeStatus),
    startedAt: pick3(snap.started_at, existing?.startedAt, at),
    toolCount: pick3(snap.tool_count, existing?.toolCount, 0),
    lastTool: pick3(snap.last_tool, existing?.lastTool, null),
    childSessionId: pick3(snap.child_session_id, existing?.childSessionId, null),
    delegationId: pick3(snap.delegation_id, existing?.delegationId, null)
  }
}

/** Merge `subagent.list` — per-session snapshots + failed async tasks. */
export function mergeSubagentList(
  state: TraceState,
  sessionId: string,
  list: WireSubagentList,
  at: number
): TraceState {
  const up = upsertSession(state, sessionId, at, {})

  for (const snap of list.subagents ?? []) {
    if (!snap.subagent_id) {
      continue
    }

    const existing = up.session.subagents.find(s => s.id === snap.subagent_id)
    upsertSubagent(up.session, nodeFromSnapshot(snap, existing, at))
  }

  for (const failed of list.delegations ?? []) {
    const id = `failed:${failed.delegation_id}:${failed.task_index ?? 0}`

    if (up.session.subagents.some(s => s.id === id)) {
      continue
    }

    upsertSubagent(up.session, {
      id,
      depth: 0,
      goal: failed.goal ?? '',
      taskIndex: failed.task_index ?? 0,
      taskCount: 1,
      status: 'failed',
      startedAt: failed.dispatched_at ?? null,
      endedAt: failed.completed_at ?? null,
      toolCount: 0,
      inputTokens: 0,
      outputTokens: 0,
      reasoningTokens: 0,
      apiCalls: 0,
      outputTail: [],
      tools: [],
      error: failed.error ?? null,
      delegationId: failed.delegation_id
    })
  }

  return replaceSession(up.state, up.session)
}

/** Hydrate persisted tool rows (`session.history` messages or canonical
 *  `session.transcript` rows — same tool fields). Live rows win: history
 *  only fills calls this window never observed. */
export function hydrateHistory(state: TraceState, sessionId: string, rows: WireHistoryRow[], at: number): TraceState {
  const up = upsertSession(state, sessionId, at, {})
  rows.forEach((row, index) => {
    if (row.role !== 'tool') {
      return
    }

    const id = row.tool_call_id || `hist-${row.row_id ?? index}`

    if (up.session.tools.some(t => t.id === id)) {
      return
    }

    const content = typeof row.content === 'string' ? row.content : typeof row.text === 'string' ? row.text : null
    upsertTool(up.session, {
      id,
      name: row.name || 'tool',
      status: 'done',
      origin: 'history',
      args: row.args ?? null,
      preview: row.context ?? null,
      labels: row.labels ?? null,
      endedAt: row.timestamp ?? null,
      resultText: content,
      resultChars: content?.length ?? null,
      raw: rawFrame(row)
    })
  })
  up.session.historyHydrated = true

  return replaceSession(up.state, up.session)
}

// ── Selectors ───────────────────────────────────────────────────────────────

export function sessionById(state: TraceState, sessionId: string | null | undefined): SessionTrace | null {
  return sessionId ? (state.sessions.find(s => s.sessionId === sessionId) ?? null) : null
}

export function findTool(state: TraceState, sel: Extract<SelectedStep, { kind: 'tool' }>): ToolCall | null {
  const session = sessionById(state, sel.sessionId)

  if (!session) {
    return null
  }

  if (sel.subagentId) {
    const node = session.subagents.find(s => s.id === sel.subagentId)

    return node?.tools.find(t => t.id === sel.toolId) ?? null
  }

  return session.tools.find(t => t.id === sel.toolId) ?? null
}

export function findSubagent(state: TraceState, sel: Extract<SelectedStep, { kind: 'subagent' }>): SubagentNode | null {
  return sessionById(state, sel.sessionId)?.subagents.find(s => s.id === sel.subagentId) ?? null
}

export function runningToolCount(state: TraceState): number {
  return state.sessions.reduce(
    (n, s) => n + s.tools.filter(t => t.status === 'running' || t.status === 'generating').length,
    0
  )
}

export function activeSubagentCount(state: TraceState): number {
  return state.sessions.reduce((n, s) => n + s.subagents.filter(a => !isTerminalSubagent(a.status)).length, 0)
}

/** Whether a session has anything live to show in the compact pane. */
export function hasLiveActivity(session: SessionTrace): boolean {
  return (
    session.turnActive ||
    session.tools.some(t => t.status === 'running' || t.status === 'generating') ||
    session.subagents.some(
      n => !isTerminalSubagent(n.status) || n.status === 'failed' || n.status === 'error' || n.status === 'timeout'
    )
  )
}

/** Sessions with any activity, most-recent first. */
export function activeSessions(state: TraceState): SessionTrace[] {
  return [...state.sessions].sort((a, b) => b.lastActivityAt - a.lastActivityAt)
}

// ── Fetch helpers (React Query feeds; merges stay pure) ─────────────────────

export const traceKeys = {
  delegation: (profile: string) => ['agent-trace', 'delegation', profile] as const,
  history: (profile: string, sessionId: string) => ['agent-trace', 'history', profile, sessionId] as const,
  subagents: (profile: string, sessionId: string) => ['agent-trace', 'subagents', profile, sessionId] as const,
  tail: (profile: string, sessionId: string, subagentId: string) =>
    ['agent-trace', 'tail', profile, sessionId, subagentId] as const
}

export async function fetchDelegationStatus(profile: string): Promise<WireDelegationStatus> {
  return host.request<WireDelegationStatus>('delegation.status', { profile })
}

/** `session.history` on the embedded gateway; the canonical gateway answers
 *  `session.transcript` instead — same tool-row fields, one fallback. */
export async function fetchSessionHistory(profile: string, sessionId: string): Promise<WireHistoryRow[]> {
  try {
    const result = await host.request<{ messages?: WireHistoryRow[] }>('session.history', {
      session_id: sessionId,
      profile
    })

    return result.messages ?? []
  } catch {
    const result = await host.request<{ rows?: WireHistoryRow[] }>('session.transcript', {
      session_id: sessionId,
      profile,
      order: 'latest',
      limit: 200
    })

    return result.rows ?? []
  }
}

export async function fetchSubagentList(profile: string, sessionId: string): Promise<WireSubagentList> {
  return host.request<WireSubagentList>('subagent.list', { session_id: sessionId, profile })
}

export async function fetchSubagentTail(
  profile: string,
  sessionId: string,
  subagentId: string
): Promise<{ available?: boolean; text?: string }> {
  return host.request<{ available?: boolean; text?: string }>('subagent.tail', {
    session_id: sessionId,
    profile,
    subagent_id: subagentId
  })
}

// ── Event wiring ────────────────────────────────────────────────────────────

const TRACED_EVENTS = [
  'tool.generating',
  'tool.start',
  'tool.progress',
  'tool.complete',
  'subagent.spawn_requested',
  'subagent.start',
  'subagent.progress',
  'subagent.tool',
  'subagent.thinking',
  'subagent.complete',
  'message.start',
  'message.complete',
  'session.usage',
  'session.info',
  'session.title',
  'error'
] as const

/** Subscribe the reducer to the gateway event tap. Disposal is tracked by
 *  the plugin context (unload/disable removes every listener). */
export function bindTrace(ctx: PluginContext): void {
  for (const type of TRACED_EVENTS) {
    ctx.onEvent(type, event => {
      if ($tracePaused.get()) {
        return
      }

      $trace.set(
        applyEvent($trace.get(), {
          type: event.type,
          at: Date.now() / 1000,
          sessionId: event.session_id,
          profile: event.profile,
          connectionId: event.connectionId,
          payload: event.payload
        })
      )
    })
  }
}
