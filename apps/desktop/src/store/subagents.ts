import type { BackendGatewayEventMap, BackendGatewayEventName, SubagentSnapshot, SubagentStatus } from '@hermes/shared'
import { atom } from 'nanostores'

import { capitalize } from '@/lib/text'

/** The six `subagent.*` notifications, straight off the generated event map. */
export type SubagentEventName = Extract<BackendGatewayEventName, `subagent.${string}`>

/** Every `subagent.*` frame carries the same generated payload. */
export type SubagentPayload = BackendGatewayEventMap[SubagentEventName]

/** `delegate.*` tags the tool-result fallback rows synthesized when a gateway
 *  relays no native `subagent.*` events. */
export type SubagentSource = 'delegate.complete' | 'delegate.running' | SubagentEventName

export type SubagentStreamKind = 'progress' | 'summary' | 'thinking' | 'tool'

export interface SubagentStreamEntry {
  at: number
  isError?: boolean
  kind: SubagentStreamKind
  text: string
}

export interface SubagentProgress {
  id: string
  parentId: null | string
  goal: string
  /** The child's own stored session id — lets UIs open its session window. */
  sessionId?: string
  /** Batch (delegation) id — exact grouping key for one fan-out's workers,
   *  so concurrent/nested batches never merge into one group. */
  delegationId?: string
  model?: string
  status: SubagentStatus
  taskCount: number
  taskIndex: number
  startedAt: number
  updatedAt: number
  durationSeconds?: number
  inputTokens?: number
  outputTokens?: number
  toolCount?: number
  filesRead: string[]
  filesWritten: string[]
  stream: SubagentStreamEntry[]
  summary?: string
  /** Active tool while running — cleared on terminal status. */
  currentTool?: string
}

export interface SubagentNode extends SubagentProgress {
  children: SubagentNode[]
}

/** The backend's terminal set (`tools/delegate_tool_child_run.py`). */
const TERMINAL: ReadonlySet<SubagentStatus> = new Set(['completed', 'error', 'failed', 'interrupted', 'timeout'])
const ACTIVE: ReadonlySet<SubagentStatus> = new Set(['queued', 'running'])
const TERMINAL_SOURCES: ReadonlySet<SubagentSource> = new Set(['delegate.complete', 'subagent.complete'])
const MAX_STREAM = 24
const PREVIEW_MAX = 220
const TOOL_PREVIEW_MAX = 96

export const $subagentsBySession = atom<Record<string, SubagentProgress[]>>({})

export const isSubagentActive = (status: SubagentStatus) => ACTIVE.has(status)

// A completion frame is terminal by definition: an emitter that still reports
// queued/running (or reports nothing) must not leave the row spinning forever.
const statusOf = (payload: SubagentPayload, prev: SubagentProgress | undefined, source?: SubagentSource) => {
  const wire = payload.status ?? prev?.status ?? 'running'

  return source && TERMINAL_SOURCES.has(source) && !TERMINAL.has(wire) ? 'failed' : wire
}

const compact = (text: string, max = PREVIEW_MAX) => {
  const line = text.replace(/\s+/g, ' ').trim()

  if (!line) {
    return ''
  }

  return line.length > max ? `${line.slice(0, max - 1)}…` : line
}

const toolLabel = (name: string) => name.split('_').filter(Boolean).map(capitalize).join(' ') || name

const formatTool = (name: string, preview = '') => {
  const snippet = compact(preview, TOOL_PREVIEW_MAX)

  return snippet ? `${toolLabel(name)}("${snippet}")` : toolLabel(name)
}

const idOf = (p: SubagentPayload) => p.subagent_id || `${p.parent_id || 'root'}:${p.task_index}:${p.goal}`

const appendStream = (stream: SubagentStreamEntry[], entry: SubagentStreamEntry) => {
  const last = stream.at(-1)

  if (last?.kind === entry.kind && last.text === entry.text && last.isError === entry.isError) {
    return stream
  }

  return [...stream, entry].slice(-MAX_STREAM)
}

// The backend sends no summary on a hard child timeout (only a preview like
// "Timed out after 612.3s" + duration_seconds). Synthesize it so the terminal
// row explains why it failed instead of rendering as a bare failure.
const timeoutSummary = (payload: SubagentPayload) =>
  payload.status === 'timeout' ? `Timed out after ${payload.duration_seconds ?? '?'}s` : ''

function streamFromPayload(
  payload: SubagentPayload,
  status: SubagentStatus,
  source: SubagentSource | undefined,
  at: number
): SubagentStreamEntry[] {
  const out: SubagentStreamEntry[] = []
  const preview = payload.tool_preview || payload.text || ''
  const text = compact(payload.text || preview)

  for (const tail of payload.output_tail ?? []) {
    const line = tail.tool ? formatTool(tail.tool, tail.preview) : compact(tail.preview)

    if (line) {
      out.push({ at, isError: tail.is_error, kind: tail.tool ? 'tool' : 'progress', text: line })
    }
  }

  if (payload.tool_name) {
    out.push({ at, kind: 'tool', text: formatTool(payload.tool_name, preview) })
  }

  if (source === 'subagent.progress' && text) {
    out.push({ at, kind: 'progress', text })
  }

  if (source === 'subagent.thinking' && text) {
    out.push({ at, kind: 'thinking', text })
  }

  const summary = compact(payload.summary || payload.text || timeoutSummary(payload))

  if (TERMINAL.has(status) && summary) {
    out.push({ at, isError: status !== 'completed', kind: 'summary', text: summary })
  }

  return out
}

/** First non-empty text, so a frame that omits a field keeps the previous row's value. */
const firstText = (...values: (null | string | undefined)[]) => values.find((value): value is string => Boolean(value))

/** First present number (0 counts), so a frame that omits a field keeps the previous row's value. */
const firstNumber = (...values: (null | number | undefined)[]) =>
  values.find((value): value is number => value !== null && value !== undefined)

/** A frame's list wins only when it says something; an empty list keeps the previous row's. */
const carryList = (next: null | string[] | undefined, prev: string[]) => (next?.length ? next : prev)

/** The row a first frame lands on: every carry-over below reads it as if a previous row existed. */
const blankProgress = (id: string, at: number): SubagentProgress => ({
  id,
  parentId: null,
  goal: '',
  status: 'queued',
  taskCount: 1,
  taskIndex: 0,
  startedAt: at,
  updatedAt: at,
  filesRead: [],
  filesWritten: [],
  stream: []
})

// Key order is load-bearing: reconcileSubagentSnapshot compares projections by
// JSON, so the roster projection below must list the same fields in order.
function toProgress(payload: SubagentPayload, prev: SubagentProgress | undefined, source?: SubagentSource) {
  const at = Date.now()
  const status = statusOf(payload, prev, source)
  const base = prev ?? blankProgress(idOf(payload), at)
  const stream = streamFromPayload(payload, status, source, at).reduce(appendStream, base.stream)

  return {
    id: base.id,
    parentId: firstText(payload.parent_id, base.parentId) ?? null,
    goal: firstText(payload.goal, base.goal) ?? 'Subagent',
    sessionId: firstText(payload.child_session_id, base.sessionId),
    delegationId: firstText(payload.delegation_id, base.delegationId),
    model: firstText(payload.model, base.model),
    status,
    taskCount: payload.task_count,
    taskIndex: payload.task_index,
    startedAt: base.startedAt,
    updatedAt: at,
    durationSeconds: firstNumber(payload.duration_seconds, base.durationSeconds),
    inputTokens: firstNumber(payload.input_tokens, base.inputTokens),
    outputTokens: firstNumber(payload.output_tokens, base.outputTokens),
    toolCount: firstNumber(payload.tool_count, base.toolCount),
    filesRead: carryList(payload.files_read, base.filesRead),
    filesWritten: carryList(payload.files_written, base.filesWritten),
    stream,
    summary: firstText(payload.summary, timeoutSummary(payload), base.summary),
    currentTool: TERMINAL.has(status) ? undefined : firstText(payload.tool_name, base.currentTool)
  } satisfies SubagentProgress
}

/** A roster row records the last tool a child ran, not a live call, and carries
 *  no stream of its own — seed cold activity only. */
function fromRosterRow(row: SubagentSnapshot, prev: SubagentProgress | undefined) {
  const startedAt = firstNumber(row.started_at ? row.started_at * 1000 : null, prev?.startedAt) ?? Date.now()
  const base = prev ?? blankProgress(row.subagent_id, startedAt)

  const seeded: SubagentStreamEntry[] = row.last_tool
    ? [{ at: base.updatedAt, kind: 'tool', text: formatTool(row.last_tool) }]
    : []

  return {
    id: row.subagent_id,
    parentId: firstText(row.parent_id, base.parentId) ?? null,
    goal: firstText(row.goal, base.goal) ?? 'Subagent',
    sessionId: base.sessionId,
    delegationId: firstText(row.delegation_id, base.delegationId),
    model: firstText(row.model, base.model),
    status: row.status,
    taskCount: base.taskCount,
    taskIndex: base.taskIndex,
    startedAt,
    updatedAt: base.updatedAt,
    durationSeconds: base.durationSeconds,
    inputTokens: base.inputTokens,
    outputTokens: base.outputTokens,
    toolCount: firstNumber(row.tool_count, base.toolCount),
    filesRead: base.filesRead,
    filesWritten: base.filesWritten,
    stream: base.stream.length ? base.stream : seeded,
    summary: base.summary,
    currentTool: TERMINAL.has(row.status) ? undefined : base.currentTool
  } satisfies SubagentProgress
}

/** Reconcile a scoped, race-checked snapshot without replacing stream history. */
export function reconcileSubagentSnapshot(sid: string, children: readonly SubagentSnapshot[]) {
  const map = $subagentsBySession.get()
  const previous = map[sid] ?? []
  const ids = new Set(children.map(row => row.subagent_id))
  const next = previous.filter(item => TERMINAL.has(item.status) || ids.has(item.id))

  for (const row of children) {
    if (!row.subagent_id) {
      continue
    }

    const index = next.findIndex(item => item.id === row.subagent_id)
    const prev = next[index]

    if (prev && TERMINAL.has(prev.status)) {
      continue
    }

    const projected = fromRosterRow(row, prev)

    if (index < 0) {
      next.push(projected)
    } else {
      next[index] = JSON.stringify(prev) === JSON.stringify(projected) ? prev : projected
    }
  }

  if (next.length !== previous.length || next.some((item, index) => item !== previous[index])) {
    $subagentsBySession.set({ ...map, [sid]: next })
  }
}

export function clearSessionSubagents(sid: string) {
  const map = $subagentsBySession.get()

  if (!(sid in map)) {
    return
  }

  const { [sid]: _drop, ...rest } = map
  $subagentsBySession.set(rest)
}

/**
 * Prune terminal-status subagent rows for a session, leaving running/queued
 * entries untouched. Used at the `message.start` boundary in the desktop
 * message-stream hook so that the *previous* turn's finished rows get flushed
 * from the display while background subagents that outlived the spawning turn
 * remain visible (and still accept late progress/complete events).
 *
 * Distinct from `clearSessionSubagents` (used by the Stop action, which
 * genuinely cancels running subagents and so should drop them all) and from
 * `pruneDelegateFallbackSubagents` (which filters by id prefix to remove
 * placeholder rows once the real native event arrives).
 */
export function pruneFinishedSessionSubagents(sid: string) {
  const map = $subagentsBySession.get()
  const list = map[sid]

  if (!list?.length) {
    return
  }

  const next = list.filter(item => isSubagentActive(item.status))

  if (next.length === list.length) {
    return
  }

  $subagentsBySession.set({ ...map, [sid]: next })
}

export function pruneDelegateFallbackSubagents(sid: string) {
  const map = $subagentsBySession.get()
  const list = map[sid]

  if (!list?.length) {
    return
  }

  const next = list.filter(item => !item.id.startsWith('delegate-tool:'))

  if (next.length === list.length) {
    return
  }

  $subagentsBySession.set({ ...map, [sid]: next })
}

export function upsertSubagent(sid: string, payload: SubagentPayload, createIfMissing = true, source?: SubagentSource) {
  const map = $subagentsBySession.get()
  const list = map[sid] ?? []
  const id = idOf(payload)
  const idx = list.findIndex(item => item.id === id)

  if (idx < 0 && !createIfMissing) {
    return
  }

  const prev = idx >= 0 ? list[idx] : undefined

  if (prev && TERMINAL.has(prev.status)) {
    return
  }

  const next = toProgress(payload, prev, source)
  const nextList = idx >= 0 ? list.map(item => (item.id === id ? next : item)) : [...list, next]

  $subagentsBySession.set({ ...map, [sid]: nextList })
}

export function buildSubagentTree(items: readonly SubagentProgress[]): SubagentNode[] {
  const nodes = new Map<string, SubagentNode>()

  for (const item of items) {
    nodes.set(item.id, { ...item, children: [] })
  }

  const roots: SubagentNode[] = []

  for (const node of nodes.values()) {
    const parent = node.parentId ? nodes.get(node.parentId) : null

    if (parent) {
      parent.children.push(node)
    } else {
      roots.push(node)
    }
  }

  const sort = (a: SubagentNode, b: SubagentNode) =>
    a.startedAt - b.startedAt || a.taskIndex - b.taskIndex || a.goal.localeCompare(b.goal)

  const walk = (node: SubagentNode) => node.children.sort(sort).forEach(walk)
  roots.sort(sort).forEach(walk)

  return roots
}

export const activeSubagentCount = (items: readonly SubagentProgress[]) =>
  items.filter(item => isSubagentActive(item.status)).length

/** Every terminal status that is not a clean completion — `error` and `timeout`
 *  are failures the backend reports under their own names. */
export const failedSubagentCount = (items: readonly SubagentProgress[]) =>
  items.filter(item => TERMINAL.has(item.status) && item.status !== 'completed').length

/** Flatten every session's subagents — the scope the Spawn-tree panel and the
 *  status-bar indicator must agree on. */
export const allSubagents = (bySession: Record<string, SubagentProgress[]>) => Object.values(bySession).flat()
