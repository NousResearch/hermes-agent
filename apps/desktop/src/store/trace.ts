import { atom } from 'nanostores'

/** Span kinds, mirroring agent/trace_builder.py (OpenInference conventions). */
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
}

export interface TraceDoc {
  traceId: string
  rootSessionId: string
  rootSpanId: null | string
  start: number
  end: number
  duration: number
  metadata: Record<string, unknown>
  spans: TraceSpan[]
}

export interface TraceTurnSummary {
  index: number
  label: string
  start: number
  end: number
  duration: number
  spanCount: number
}

export interface TraceSpanNode extends TraceSpan {
  depth: number
  children: TraceSpanNode[]
}

/** Raw wire payloads from the gateway (snake_case). */
interface WireSpan {
  span_id?: string
  parent_id?: null | string
  name?: string
  kind?: string
  start?: number
  end?: number
  duration?: number
  status?: string
  session_id?: null | string
  attributes?: Record<string, unknown>
}

interface WireTrace {
  trace_id?: string
  root_session_id?: string
  root_span_id?: null | string
  start?: number
  end?: number
  duration?: number
  metadata?: Record<string, unknown>
  spans?: WireSpan[]
}

const asKind = (v: unknown): TraceSpanKind =>
  v === 'AGENT' || v === 'CHAIN' || v === 'LLM' || v === 'TOOL' ? v : 'CHAIN'

const asStatus = (v: unknown): TraceSpanStatus => (v === 'ok' || v === 'error' ? v : 'unset')

const num = (v: unknown, fallback = 0) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback)

function toSpan(w: WireSpan): TraceSpan {
  const start = num(w.start)
  const end = num(w.end, start)

  return {
    id: String(w.span_id ?? ''),
    parentId: w.parent_id ?? null,
    name: String(w.name ?? ''),
    kind: asKind(w.kind),
    start,
    end,
    duration: num(w.duration, Math.max(0, end - start)),
    status: asStatus(w.status),
    sessionId: w.session_id ?? null,
    attributes: w.attributes ?? {}
  }
}

export function toTraceDoc(wire: WireTrace): TraceDoc {
  const spans = (wire.spans ?? []).map(toSpan)

  return {
    traceId: String(wire.trace_id ?? ''),
    rootSessionId: String(wire.root_session_id ?? ''),
    rootSpanId: wire.root_span_id ?? null,
    start: num(wire.start),
    end: num(wire.end),
    duration: num(wire.duration),
    metadata: wire.metadata ?? {},
    spans
  }
}

/**
 * Shift every timestamp in a trace by a constant so its root starts at
 * `newStart`, preserving all relative durations. Used to fold a settled DB
 * trace (server epoch) onto the live turn's client epoch, so swapping live →
 * DB lands on the same on-screen window — the time-rebase half of an A→B
 * hand-off where B's exact spans replace A's approximate ones in place.
 */
export function rebaseTrace(doc: TraceDoc, newStart: number): TraceDoc {
  const delta = newStart - doc.start

  if (!Number.isFinite(delta) || delta === 0) {
    return doc
  }

  return {
    ...doc,
    start: doc.start + delta,
    end: doc.end + delta,
    spans: doc.spans.map(s => ({ ...s, start: s.start + delta, end: s.end + delta }))
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

export const $trace = atom<TraceDoc | null>(null)
export const $traceTurns = atom<TraceTurnSummary[]>([])
export const $traceLoading = atom<boolean>(false)
export const $traceError = atom<null | string>(null)
export const $selectedSpanId = atom<null | string>(null)
export const $hoveredSpanId = atom<null | string>(null)
export const $traceLabelsCollapsed = atom<boolean>(false)

/** Which turn the agents overlay shows: 'latest' follows the newest turn (and
 *  the live stream), 'all' is the whole session, a number pins a settled turn. */
export type TraceSelection = 'all' | 'latest' | number
export const $traceSelection = atom<TraceSelection>('latest')

/** Clear hover only if this span is the current one (avoids enter/leave races). */
export function clearHoveredSpan(id: string) {
  if ($hoveredSpanId.get() === id) {
    $hoveredSpanId.set(null)
  }
}

interface WireTurn {
  index?: number
  label?: string
  start?: number
  end?: number
  duration?: number
  span_count?: number
}

export function toTurnSummaries(wire: { turns?: WireTurn[] }): TraceTurnSummary[] {
  return (wire.turns ?? []).map(t => ({
    index: num(t.index),
    label: String(t.label ?? `turn ${num(t.index)}`),
    start: num(t.start),
    end: num(t.end),
    duration: num(t.duration),
    spanCount: num(t.span_count)
  }))
}

export function setTrace(trace: null | TraceDoc) {
  $trace.set(trace)
  $selectedSpanId.set(null)
}
