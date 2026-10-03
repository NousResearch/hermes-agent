import type {
  RelayEvent,
  TraceDoc,
  TracePhase,
  TraceSpan,
  TraceSpanKind,
  TraceSpanStatus,
  TraceTurnSummary
} from '@/store/trace'

// Scope names Hermes gives its own Relay scopes (agent/relay_runtime.py).
const SESSION_SCOPE = 'hermes.session'
const TURN_SCOPE = 'hermes.turn'
const LOGICAL_LLM_SCOPE = 'hermes.logical_llm_call'
const TURN_INPUT_MARK = 'hermes.turn.input'
// Streamed model output (hermes_cli/observability/relay_traces.py): Relay marks attach only to
// scopes, so they sit on the call's parent scope and name the call by `llm_uuid`.
const FIRST_TOKEN_MARK = 'hermes.llm.first_token'
const STREAM_MARK = 'hermes.llm.stream'
const SPAWNED_BY_KEY = 'hermes.spawned_by_tool_call_id'
const DELEGATE_TOOL = 'delegate_task'

export interface TraceLabels {
  llmCall: string
  session: string
  subagent: string
  turn: string
}

const KIND_BY_CATEGORY: Record<string, TraceSpanKind> = { agent: 'AGENT', llm: 'LLM', tool: 'TOOL' }

type Json = Record<string, unknown>

const obj = (value: unknown): Json =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Json) : {}

const str = (value: unknown): string => (typeof value === 'string' ? value : '')

const num = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined

export function relayEventSeconds(event: RelayEvent): number {
  const ms = Date.parse(event.timestamp)

  return Number.isFinite(ms) ? ms / 1000 : 0
}

function pretty(value: unknown): string {
  if (value === null || value === undefined || value === '') {
    return ''
  }

  return typeof value === 'string' ? value : JSON.stringify(value, null, 2)
}

function spanStatus(end: RelayEvent | undefined): TraceSpanStatus {
  if (!end) {
    return 'running'
  }

  const outcome = str(obj(end.data).outcome)

  if (obj(end.metadata)['otel.status_code'] === 'ERROR' || outcome === 'failed') {
    return 'error'
  }

  return outcome === 'cancelled' || obj(end.metadata)['hermes.status'] === 'abandoned' ? 'unset' : 'ok'
}

/** What the inspector shows: the model, tokens and payload Relay recorded for the span. */
function spanAttributes(start: RelayEvent, end: RelayEvent | undefined): Record<string, unknown> {
  const attributes: Record<string, unknown> = {}
  const profile = { ...obj(start.category_profile), ...obj(end?.category_profile) }

  if (start.category === 'tool') {
    attributes['tool.name'] = start.name
    attributes['tool.call_id'] = str(profile.tool_call_id)
    attributes['input.value'] = pretty(start.data)
    attributes['output.value'] = pretty(end?.data)

    return attributes
  }

  if (start.category === 'llm') {
    const request = obj(obj(start.data).content)
    const response = obj(end?.data)
    const annotated = obj(profile.annotated_response)
    const usage = { ...obj(response.usage), ...obj(annotated.usage) }
    const messages = Array.isArray(request.messages) ? request.messages : []
    const last = obj(messages.at(-1))
    const choice = obj(Array.isArray(response.choices) ? response.choices[0] : undefined)

    attributes['llm.model_name'] = str(profile.model_name) || str(request.model)
    attributes['llm.token_count.prompt'] = num(usage.prompt_tokens) ?? num(usage.input_tokens)
    attributes['llm.token_count.completion'] = num(usage.completion_tokens) ?? num(usage.output_tokens)
    // Relay's normalized usage has no reasoning field; read the provider's own usage shapes.
    attributes['llm.token_count.reasoning'] =
      num(usage.reasoning_tokens) ??
      num(obj(usage.completion_tokens_details).reasoning_tokens) ??
      num(obj(usage.output_tokens_details).reasoning_tokens)
    // Relay's finish-reason enum is lossy; prefer what the provider actually said.
    attributes['hermes.finish_reason'] =
      str(response.finish_reason) ||
      str(choice.finish_reason) ||
      str(response.stop_reason) ||
      str(annotated.finish_reason)
    attributes['input.value'] = pretty(last.content ?? request.messages)
    attributes['output.value'] = pretty(annotated.message ?? response.content ?? end?.data)

    return attributes
  }

  const metadata = obj(start.metadata)
  attributes['session.source'] = str(metadata['hermes.execution_surface'])

  return attributes
}

/**
 * Fold one session's Relay event log into the waterfall's span tree. A span is a
 * scope start/end pair (same uuid); an open scope is running and extends to `nowSec`.
 * A delegated session's scope hangs off the turn in Relay; it is drawn under the
 * `delegate_task` call that was open when it started, which is the call that spawned it.
 */
export function foldRelayEvents(events: RelayEvent[], labels: TraceLabels, nowSec: number): TraceDoc {
  const starts = new Map<string, RelayEvent>()
  const ends = new Map<string, RelayEvent>()
  const turnInputs = new Map<string, string>()
  const streamMarks: RelayEvent[] = []

  for (const event of events) {
    if (event.kind === 'mark') {
      if (event.name === TURN_INPUT_MARK && event.parent_uuid) {
        turnInputs.set(event.parent_uuid, str(obj(event.data).preview))
      } else if (event.name === FIRST_TOKEN_MARK || event.name === STREAM_MARK) {
        streamMarks.push(event)
      }
    } else if (event.scope_category === 'end') {
      ends.set(event.uuid, event)
    } else {
      starts.set(event.uuid, event)
    }
  }

  const spans: TraceSpan[] = []
  const sessionOf = new Map<string, null | string>()

  const owningSession = (uuid: null | string | undefined): null | string => {
    if (!uuid || !starts.has(uuid)) {
      return null
    }

    const cached = sessionOf.get(uuid)

    if (cached !== undefined) {
      return cached
    }

    const start = starts.get(uuid)!
    const own = start.name === SESSION_SCOPE ? str(obj(start.metadata)['hermes.session_id']) || null : null
    const resolved = own ?? owningSession(start.parent_uuid)
    sessionOf.set(uuid, resolved)

    return resolved
  }

  for (const [uuid, start] of starts) {
    const end = ends.get(uuid)
    const began = relayEventSeconds(start)
    const finished = end ? Math.max(began, relayEventSeconds(end)) : Math.max(began, nowSec)
    const metadata = obj(start.metadata)
    const isSubagent = start.name === SESSION_SCOPE && Boolean(metadata['hermes.parent_session_id'])

    const name =
      start.name === SESSION_SCOPE
        ? isSubagent
          ? labels.subagent
          : labels.session
        : start.name === TURN_SCOPE
          ? turnInputs.get(uuid) || labels.turn
          : start.name === LOGICAL_LLM_SCOPE
            ? labels.llmCall
            : start.category === 'llm'
              ? str(obj(start.category_profile).model_name) || start.name
              : start.name

    spans.push({
      id: uuid,
      parentId: start.parent_uuid && starts.has(start.parent_uuid) ? start.parent_uuid : null,
      name,
      kind: KIND_BY_CATEGORY[start.category ?? ''] ?? 'CHAIN',
      start: began,
      end: finished,
      duration: finished - began,
      status: spanStatus(end),
      sessionId: owningSession(uuid),
      attributes: spanAttributes(start, end)
    })
  }

  const delegatesByParent = new Map<null | string, TraceSpan[]>()

  for (const span of spans) {
    if (span.kind === 'TOOL' && span.attributes['tool.name'] === DELEGATE_TOOL) {
      delegatesByParent.set(span.parentId, [...(delegatesByParent.get(span.parentId) ?? []), span])
    }
  }

  const toolByCallId = new Map(
    spans.filter(s => s.kind === 'TOOL' && s.attributes['tool.call_id']).map(s => [s.attributes['tool.call_id'], s])
  )

  for (const span of spans) {
    if (span.kind === 'AGENT' && span.parentId) {
      // Exact when the child names its spawning call; older traces fall back to the delegate
      // call that was open when the child started.
      const spawnedBy = str(obj(starts.get(span.id)?.metadata)[SPAWNED_BY_KEY])

      const spawner =
        (spawnedBy && toolByCallId.get(spawnedBy)) ||
        delegatesByParent.get(span.parentId)?.find(d => d.start <= span.start && span.start <= d.end)

      if (spawner) {
        span.parentId = spawner.id
      }
    }
  }

  attachStreams(spans, streamMarks)
  settleOpenSessions(spans, starts, ends)

  const start = spans.length ? Math.min(...spans.map(s => s.start)) : nowSec
  const end = spans.length ? Math.max(...spans.map(s => s.end)) : nowSec
  const root = spans.find(s => s.parentId === null && s.kind === 'AGENT')

  return { rootSessionId: root?.sessionId ?? '', start, end, duration: end - start, spans }
}

/**
 * Hang each streamed-output mark on its model call and derive what the bar and inspector show:
 * the wait before the first token, then reasoning/text phases, and the text streamed so far.
 */
function attachStreams(spans: TraceSpan[], marks: RelayEvent[]) {
  if (marks.length === 0) {
    return
  }

  const byId = new Map(spans.map(span => [span.id, span]))
  const llmsByParent = new Map<string, TraceSpan[]>()

  for (const span of spans) {
    if (span.kind === 'LLM' && span.parentId) {
      llmsByParent.set(span.parentId, [...(llmsByParent.get(span.parentId) ?? []), span])
    }
  }

  const marksBySpan = new Map<TraceSpan, RelayEvent[]>()

  for (const mark of marks) {
    const at = relayEventSeconds(mark)
    const named = byId.get(str(obj(mark.data).llm_uuid))

    // A managed call's mark names only its request: take the newest attempt already running.
    const target =
      named ??
      (llmsByParent.get(mark.parent_uuid ?? '') ?? [])
        .filter(span => span.start <= at)
        .reduce<TraceSpan | undefined>(
          (latest, span) => (!latest || span.start > latest.start ? span : latest),
          undefined
        )

    if (target) {
      marksBySpan.set(target, [...(marksBySpan.get(target) ?? []), mark])
    }
  }

  for (const [span, spanMarks] of marksBySpan) {
    spanMarks.sort((a, b) => relayEventSeconds(a) - relayEventSeconds(b))
    const first = spanMarks.find(mark => mark.name === FIRST_TOKEN_MARK)
    const phases: TracePhase[] = []
    const text: Record<string, string> = { reasoning: '', text: '' }

    const push = (kind: TracePhase['kind'], from: number, to: number) => {
      const start = Math.max(span.start, Math.min(from, span.end))
      const end = Math.max(start, Math.min(to, span.end))
      const last = phases.at(-1)

      if (last?.kind === kind) {
        last.end = end
      } else if (end > start) {
        phases.push({ kind, start, end })
      }
    }

    let cursor = first ? relayEventSeconds(first) : span.start
    let kind: TracePhase['kind'] = str(obj(first?.data).kind) === 'reasoning' ? 'reasoning' : 'text'

    if (first) {
      push('wait', span.start, cursor)
      span.attributes['llm.first_token_s'] = cursor - span.start
    }

    for (const mark of spanMarks) {
      if (mark.name !== STREAM_MARK) {
        continue
      }

      const data = obj(mark.data)
      kind = str(data.kind) === 'reasoning' ? 'reasoning' : 'text'
      text[kind] += str(data.text)
      push(kind, cursor, relayEventSeconds(mark))
      cursor = relayEventSeconds(mark)
    }

    push(kind, cursor, span.end)
    span.phases = phases

    if (text.reasoning) {
      span.attributes['llm.reasoning.value'] = text.reasoning
    }

    if (span.status === 'running' && text.text) {
      span.attributes['output.value'] = text.text
    }
  }
}

/**
 * A session scope stays open for the whole conversation, so "open" is not "working":
 * between turns it ends where its last child ended and is only running while a child is.
 */
function settleOpenSessions(spans: TraceSpan[], starts: Map<string, RelayEvent>, ends: Map<string, RelayEvent>) {
  const children = new Map<string, TraceSpan[]>()

  for (const span of spans) {
    if (span.parentId) {
      children.set(span.parentId, [...(children.get(span.parentId) ?? []), span])
    }
  }

  const settle = (span: TraceSpan): { end: number; running: boolean } => {
    const kids = (children.get(span.id) ?? []).map(settle)
    const open = !ends.has(span.id) && starts.get(span.id)?.name === SESSION_SCOPE

    if (open) {
      span.end = Math.max(span.start, ...kids.map(k => k.end))
      span.duration = span.end - span.start
      span.status = kids.some(k => k.running) ? 'running' : 'unset'
    }

    return { end: span.end, running: span.status === 'running' || kids.some(k => k.running) }
  }

  for (const span of spans) {
    if (span.parentId === null) {
      settle(span)
    }
  }
}

/** The root session's turns, oldest first. */
export function traceTurns(doc: TraceDoc): TraceTurnSummary[] {
  const roots = new Set(doc.spans.filter(s => s.parentId === null).map(s => s.id))

  return doc.spans
    .filter(s => s.parentId !== null && roots.has(s.parentId) && s.kind === 'CHAIN')
    .sort((a, b) => a.start - b.start)
    .map((turn, index) => ({
      id: turn.id,
      index,
      label: turn.name,
      start: turn.start,
      end: turn.end,
      duration: turn.duration,
      running: turn.status === 'running'
    }))
}

/** One turn as its own trace: the turn span becomes the root of its subtree. */
export function turnTrace(doc: TraceDoc, turnId: string): TraceDoc {
  const children = new Map<string, TraceSpan[]>()

  for (const span of doc.spans) {
    if (span.parentId) {
      children.set(span.parentId, [...(children.get(span.parentId) ?? []), span])
    }
  }

  const turn = doc.spans.find(s => s.id === turnId)

  if (!turn) {
    return { ...doc, spans: [] }
  }

  const spans: TraceSpan[] = [{ ...turn, parentId: null }]

  for (let i = 0; i < spans.length; i++) {
    spans.push(...(children.get(spans[i]!.id) ?? []))
  }

  return { ...doc, start: turn.start, end: turn.end, duration: turn.duration, spans }
}
