import { atom } from 'nanostores'

/** One NeMo Relay ATOF event as Hermes recorded it (payloads bounded, shape untouched). */
export interface RelayEvent {
  atof_version?: string
  category?: null | string
  category_profile?: null | Record<string, unknown>
  data?: unknown
  kind: 'mark' | 'scope'
  metadata?: null | Record<string, unknown>
  name: string
  parent_uuid?: null | string
  scope_category?: 'end' | 'start'
  timestamp: string
  uuid: string
}

/** Span kinds of the waterfall, one per Relay scope category it draws. */
export type TraceSpanKind = 'AGENT' | 'CHAIN' | 'LLM' | 'TOOL'
export type TraceSpanStatus = 'error' | 'ok' | 'running' | 'unset'

export interface TraceSpan {
  id: string
  parentId: null | string
  name: string
  kind: TraceSpanKind
  /** Epoch seconds. */
  start: number
  end: number
  duration: number
  status: TraceSpanStatus
  sessionId: null | string
  attributes: Record<string, unknown>
  /** A model call's streamed phases: waiting for the first token, then reasoning/text. */
  phases?: TracePhase[]
}

export interface TracePhase {
  kind: 'reasoning' | 'text' | 'wait'
  /** Epoch seconds. */
  start: number
  end: number
}

export interface TraceDoc {
  rootSessionId: string
  start: number
  end: number
  duration: number
  spans: TraceSpan[]
}

export interface TraceTurnSummary {
  id: string
  index: number
  label: string
  start: number
  end: number
  duration: number
  running: boolean
}

export interface TraceSpanNode extends TraceSpan {
  depth: number
  children: TraceSpanNode[]
}

/** The Relay event log of the session the Agents view is tracing: fetched once, then grown live. */
export interface TraceLog {
  /** The gateway session the log was requested for (live id or stored id). */
  sessionId: string
  /** The stored session every event rolls up to. */
  storedId: string
  events: RelayEvent[]
  recording: boolean
  truncated: boolean
}

export const $traceLog = atom<null | TraceLog>(null)
export const $traceLoading = atom<boolean>(false)
export const $traceError = atom<null | string>(null)
export const $selectedSpanId = atom<null | string>(null)
export const $hoveredSpanId = atom<null | string>(null)
export const $traceLabelsCollapsed = atom<boolean>(false)

/** Which face the Agents panel shows: the Relay trace waterfall or the live spawn tree. */
export type AgentsPanelView = 'trace' | 'tree'
export const $agentsPanelView = atom<AgentsPanelView>('trace')

/** Which turn the agents overlay shows: 'latest' follows the newest turn (and
 *  the live stream), 'all' is the whole session, a number pins a turn. */
export type TraceSelection = 'all' | 'latest' | number
export const $traceSelection = atom<TraceSelection>('latest')

/** Clear hover only if this span is the current one (avoids enter/leave races). */
export function clearHoveredSpan(id: string) {
  if ($hoveredSpanId.get() === id) {
    $hoveredSpanId.set(null)
  }
}

const eventKey = (event: RelayEvent) => `${event.uuid}:${event.scope_category ?? event.kind}`

/** Append live events to the traced log, skipping any the fetch already returned. `storedId`
 *  is the root the backend attributed them to; a compression rotation moves it forward. */
export function appendTraceEvents(sessionId: string, storedId: string, incoming: RelayEvent[]) {
  const log = $traceLog.get()

  if (!log || log.sessionId !== sessionId || incoming.length === 0) {
    return
  }

  const seen = new Set(log.events.map(eventKey))
  const fresh = incoming.filter(event => !seen.has(eventKey(event)))

  if (fresh.length > 0 || (storedId && storedId !== log.storedId)) {
    $traceLog.set({ ...log, storedId: storedId || log.storedId, events: [...log.events, ...fresh] })
  }
}

/** Pre-order flatten of the span tree with depth, sorted by start time. */
export function flattenSpanTree(trace: TraceDoc): TraceSpanNode[] {
  const byParent = new Map<null | string, TraceSpan[]>()

  for (const span of trace.spans) {
    const list = byParent.get(span.parentId) ?? []
    list.push(span)
    byParent.set(span.parentId, list)
  }

  for (const list of byParent.values()) {
    list.sort((a, b) => a.start - b.start || a.id.localeCompare(b.id))
  }

  const out: TraceSpanNode[] = []

  const walk = (parentId: null | string, depth: number) => {
    for (const span of byParent.get(parentId) ?? []) {
      const node: TraceSpanNode = { ...span, depth, children: [] }
      out.push(node)
      walk(span.id, depth + 1)
    }
  }

  walk(null, 0)

  return out
}
