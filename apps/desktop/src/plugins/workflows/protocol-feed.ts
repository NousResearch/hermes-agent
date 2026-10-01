// The activity feed's projection of the same stream — one human-readable line
// per event worth reading, and nothing for the infra chatter the timeline
// already shows as ticks.

import { type EventType, HERMES_EXT, type ProtoEvent } from './protocol'

export interface FeedLine {
  step: string
  /** Colours the line's dot in the feed. The kind IS the visual marker — the
   *  feed used to also carry a per-line glyph, which was redundant with it. */
  kind: 'start' | 'tool' | 'ok' | 'fail' | 'loop' | 'data'
  msg: string
  ext: boolean
  ts: number
}

/** Infra chatter the timeline shows as ticks and the feed leaves out. */
const FEED_HIDDEN: ReadonlySet<EventType> = new Set<EventType>([
  'TokenUsage',
  'SnapshotCaptured',
  'FrameCommitted',
  'NodePending',
  'TodoUpdated',
  'TaskOutput',
  'NodeFinished'
])

export function feedLine(e: ProtoEvent): FeedLine | null {
  if (FEED_HIDDEN.has(e.type)) {
    return null
  }

  const ext = HERMES_EXT.has(e.type)
  const base = { ext, ts: e.ts }

  switch (e.type) {
    case 'RunStarted':
      return { ...base, step: 'run', kind: 'start', msg: e.payload.scenario }

    case 'RunFinished':
      return { ...base, step: 'run', kind: 'ok', msg: `run ${e.payload.state}` }

    case 'RunPaused':
      return { ...base, step: 'run', kind: 'data', msg: 'paused' }

    case 'NodeStarted':
      return {
        ...base,
        step: e.payload.nodeId,
        kind: 'start',
        msg: `delegate_task spawned · ${e.payload.input}`
      }
    case 'AgentTraceEvent': {
      const t = e.payload.tool

      return {
        ...base,
        step: e.payload.nodeId,
        kind: 'tool',
        msg: `${t.name}${t.arg ? ` · ${t.arg}` : ''}`
      }
    }

    case 'AgentTraceSummary':
      return {
        ...base,
        step: e.payload.nodeId,
        kind: e.payload.verdict === 'FAIL' ? 'fail' : 'ok',
        msg: e.payload.summary
      }

    case 'NodeFailed':
      return {
        ...base,
        step: e.payload.nodeId,
        kind: 'fail',
        msg: e.payload.error
      }

    case 'NodeSkipped':
      return {
        ...base,
        step: e.payload.nodeId,
        kind: 'data',
        msg: `skipped · ${e.payload.reason}`
      }

    case 'GateEvaluated':
      return {
        ...base,
        step: e.payload.nodeId,
        kind: e.payload.decision === 'pass' ? 'ok' : 'loop',
        msg: e.payload.summary
      }

    case 'LoopAdvanced':
      return {
        ...base,
        step: e.payload.loopId,
        kind: 'loop',
        msg: `take ${e.payload.iteration + 1} → ${e.payload.to} · ${e.payload.feedback}`
      }

    case 'HumanWaiting':
      return {
        ...base,
        step: e.payload.nodeId,
        kind: 'data',
        msg: `waiting on you · ${e.payload.prompt}`
      }

    case 'HumanResponded':
      return {
        ...base,
        step: e.payload.nodeId,
        kind: e.payload.decision === 'approved' ? 'ok' : 'fail',
        msg: `${e.payload.decision} · ${e.payload.by}`
      }

    case 'WaitStarted':
      return {
        ...base,
        step: e.payload.nodeId,
        kind: 'data',
        msg: `waiting on ${e.payload.label}`
      }

    case 'WaitResolved':
      return { ...base, step: e.payload.nodeId, kind: 'ok', msg: e.payload.by }

    case 'UserAsk':
      return { ...base, step: e.payload.nodeId, kind: 'data', msg: e.payload.prompt }

    default:
      return null
  }
}
