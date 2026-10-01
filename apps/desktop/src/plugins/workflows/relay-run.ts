// A run's history is its Relay trace. What the runner decided is a `hermes.workflow.<Type>` mark
// on the run's session; an agent step is a child session (`wf-<runId>-<nodeId>`) whose tool spans
// are what a card's ticker shows. This folds the raw trace into the canvas's event vocabulary
// (`protocol.ts`), so the reducer stays the engine-shaped fold it is and nothing keeps a second log.

import type { ProtoEvent } from './protocol'

export interface RelayEvent {
  kind?: string
  scope_category?: string | null
  category?: string | null
  name?: string
  uuid?: string
  parent_uuid?: string | null
  timestamp?: string
  data?: unknown
  metadata?: Record<string, unknown> | null
}

const MARK = 'hermes.workflow.'
const SESSION_SCOPE = 'hermes.session'
const SESSION_ID = 'hermes.session_id'

/** One copy per (uuid, phase): a live event can arrive after a fetch that already had it. */
export const relayKey = (e: RelayEvent) => `${e.uuid ?? ''}:${e.scope_category ?? e.kind ?? ''}`

export function mergeRelay(into: RelayEvent[], incoming: RelayEvent[]): RelayEvent[] {
  const seen = new Set(into.map(relayKey))
  const fresh: RelayEvent[] = []

  for (const e of incoming) {
    const key = relayKey(e)

    if (!seen.has(key)) {
      seen.add(key)
      fresh.push(e)
    }
  }

  return fresh.length ? [...into, ...fresh] : into
}

/** The runner's own event, if this is one of its marks. */
export function markOf(e: RelayEvent): { type: ProtoEvent['type']; payload: Record<string, unknown> } | null {
  if (e.kind !== 'mark' || !e.name?.startsWith(MARK) || !e.data || typeof e.data !== 'object') {
    return null
  }

  const data = e.data as { payload?: Record<string, unknown> }

  return { type: e.name.slice(MARK.length) as ProtoEvent['type'], payload: data.payload ?? {} }
}

function preview(data: unknown): string {
  if (typeof data === 'string') {
    return data.slice(0, 80)
  }

  if (data && typeof data === 'object') {
    for (const value of Object.values(data as Record<string, unknown>)) {
      if (typeof value === 'string' && value) {
        return value.slice(0, 80)
      }

      if (value && typeof value === 'object') {
        const inner = preview(value)

        if (inner) {
          return inner
        }
      }
    }
  }

  return ''
}

const time = (e: RelayEvent) => (e.timestamp ? Date.parse(e.timestamp) : 0)

export function protoEvents(runId: string, trace: RelayEvent[]): ProtoEvent[] {
  const ordered = [...trace].sort((a, b) => time(a) - time(b))
  const parentOf = new Map<string, string>()
  const sessionOf = new Map<string, string>()

  for (const e of ordered) {
    if (e.uuid && e.parent_uuid) {
      parentOf.set(e.uuid, e.parent_uuid)
    }

    const sid = e.metadata?.[SESSION_ID]

    if (e.name === SESSION_SCOPE && e.uuid && typeof sid === 'string') {
      sessionOf.set(e.uuid, sid)
    }
  }

  const ownerOf = (uuid: string | null | undefined): string | undefined => {
    for (let at = uuid ?? undefined, hops = 0; at && hops < 64; at = parentOf.get(at), hops++) {
      const sid = sessionOf.get(at)

      if (sid) {
        return sid
      }
    }

    return undefined
  }

  const stepPrefix = `wf-${runId}-`
  const takes = new Map<string, number>()
  const out: ProtoEvent[] = []

  for (const e of ordered) {
    const mark = markOf(e)
    const ts = time(e)

    if (mark) {
      if (mark.type === 'NodeStarted') {
        takes.set(String(mark.payload.nodeId), Number(mark.payload.iteration ?? 0))
      }

      out.push({ payload: mark.payload, runId, seq: out.length, ts, type: mark.type } as ProtoEvent)

      continue
    }

    if (e.kind === 'scope' && e.scope_category === 'start' && e.category === 'tool') {
      const sid = ownerOf(e.parent_uuid)

      if (sid?.startsWith(stepPrefix)) {
        const nodeId = sid.slice(stepPrefix.length)

        out.push({
          payload: {
            iteration: takes.get(nodeId) ?? 0,
            nodeId,
            tool: { arg: preview(e.data), name: e.name ?? 'tool' }
          },
          runId,
          seq: out.length,
          ts,
          type: 'AgentTraceEvent'
        })
      }
    }
  }

  return out
}
