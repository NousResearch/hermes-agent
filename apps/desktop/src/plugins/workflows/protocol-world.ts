// Folding the event stream into the world the canvas renders.
//
// Replaying a prefix is the whole time-travel mechanism — there is no separate
// historical store, and no state here that isn't derivable from the events in
// `protocol.ts` plus the shape of what's being run.

import {
  type Checkpoint,
  type EdgeState,
  freshRuntime,
  isBrokenReply,
  type NodeRef,
  type ProtoEvent,
  type RunPhase,
  type RunShape,
  type StepRuntime,
  type StepStatus,
  type World
} from './protocol'

function initialRuntime(shape: RunShape): Record<string, StepRuntime> {
  return Object.fromEntries(shape.steps.map(id => [id, freshRuntime()]))
}

function initialEdgeState(shape: RunShape): Record<string, EdgeState> {
  return Object.fromEntries(shape.edges.map(e => [e.id, 'idle' as EdgeState]))
}

/** All checkpoints present in a stream (the scrubber's stops). */
export function checkpointsOf(events: ProtoEvent[]): Checkpoint[] {
  const out: Checkpoint[] = []
  events.forEach((e, at) => {
    if (e.type === 'FrameCommitted') {
      out.push({ no: e.payload.frameNo, label: e.payload.label, at })
    }
  })

  return out
}

/**
 * Fold `count` events into the world the canvas renders. Replaying a prefix is
 * the whole time-travel mechanism — there is no separate historical store.
 */
export function reduceEvents(events: ProtoEvent[], shape: RunShape, count = events.length): World {
  const steps = initialRuntime(shape)
  const edges = initialEdgeState(shape)
  let phase: RunPhase = 'idle'
  let take = 0
  let clockTs: number | null = null
  // A loop-back edge is lit from the moment the run advances to the next take
  // until its target actually restarts. The engine reports the bump; the edge
  // is ours.
  let loopTarget: string | null = null

  for (let i = 0; i < Math.min(count, events.length); i++) {
    const e = events[i]
    clockTs = e.ts
    const p = e.payload as Partial<NodeRef>
    const rt = p.nodeId ? steps[p.nodeId] : undefined

    switch (e.type) {
      case 'RunStarted':
        phase = 'running'
        take = 1

        break

      case 'RunFinished':
        phase = 'done'

        break

      case 'RunPaused':
        break

      case 'NodePending':
        if (rt) {
          Object.assign(rt, freshRuntime(), {
            status: 'queued' as StepStatus,
            take: e.payload.iteration + 1
          })
        }

        break

      case 'NodeStarted':
        if (rt) {
          rt.status = e.payload.loop ? 'looping' : 'running'
          rt.startedAt = e.ts
          rt.durationMs = null
          rt.input = e.payload.input
          rt.maxIters = e.payload.maxIters
          rt.take = e.payload.iteration + 1
          rt.skipped = null
        }

        if (loopTarget === e.payload.nodeId) {
          loopTarget = null
        }

        take = Math.max(take, e.payload.iteration + 1)

        break

      case 'AgentTraceEvent':
        if (rt) {
          rt.currentTool = e.payload.tool
          rt.toolCalls = [...rt.toolCalls, e.payload.tool]
          rt.iterations += 1
        }

        break

      case 'TokenUsage':
        if (rt) {
          rt.tokens += e.payload.tokens
        }

        break

      case 'TodoUpdated':
        if (rt) {
          rt.todos = e.payload.todos.map(t => ({ ...t }))
        }

        break

      case 'AgentTraceSummary':
        if (rt) {
          rt.summary = e.payload.summary
          rt.verdict = e.payload.verdict ?? null
        }

        break

      case 'TaskOutput':
        if (rt) {
          rt.output = e.payload.output
        }

        break

      case 'NodeFinished':
        if (rt) {
          // The wire says "finished" whether the agent approved or rejected —
          // a FAIL is a value in its structured output, not a task error. The
          // canvas is what turns a rejecting validator red.
          //
          // A transport error returned as the reply (auth missing, HTTP 4xx)
          // is not a verdict — treat it as the step breaking.
          rt.status = rt.verdict === 'FAIL' || isBrokenReply(rt.summary) ? 'failed' : 'done'
          rt.currentTool = null
          rt.durationMs = rt.startedAt != null ? e.ts - rt.startedAt : null
        }

        break

      case 'NodeFailed':
        if (rt) {
          rt.status = 'failed'
          rt.currentTool = null
          rt.summary = e.payload.error
          rt.durationMs = rt.startedAt != null ? e.ts - rt.startedAt : null
        }

        break

      case 'NodeSkipped':
        // Held over, not re-run — it keeps the take number and telemetry of
        // the take that satisfied it.
        if (rt) {
          rt.skipped = e.payload.reason
        }

        break

      case 'GateEvaluated':
        if (rt) {
          rt.verdict = e.payload.decision === 'pass' ? 'PASS' : 'FAIL'
          rt.summary = e.payload.summary
          rt.input = e.payload.inputs.map(v => `${v.nodeId} ${v.verdict ?? '—'}`).join(' · ')
          rt.status = e.payload.decision === 'pass' ? 'done' : 'looping'
          rt.currentTool = null
          rt.durationMs = rt.startedAt != null ? e.ts - rt.startedAt : null
        }

        break

      case 'HumanWaiting':
        // The run parks. startedAt keeps ticking — elapsed-while-blocked is
        // the one number a waiting card owes you.
        if (rt) {
          rt.status = 'waiting'
          rt.summary = e.payload.prompt

          if (rt.startedAt == null) {
            rt.startedAt = e.ts
          }
        }

        break

      case 'WaitStarted':
        if (rt) {
          rt.status = 'waiting'
          rt.input = e.payload.until
          rt.summary = e.payload.label
          rt.take = e.payload.iteration + 1

          if (rt.startedAt == null) {
            rt.startedAt = e.ts
          }
        }

        break

      case 'WaitResolved':
        if (rt) {
          // A wait has no opinion, so it reports no verdict — it either came
          // back or the run is still sitting on it.
          rt.status = 'done'
          rt.summary = e.payload.by
          rt.durationMs = rt.startedAt != null ? e.ts - rt.startedAt : null
        }

        break

      case 'HumanResponded':
        if (rt) {
          rt.verdict = e.payload.decision === 'approved' ? 'PASS' : 'FAIL'
          rt.status = e.payload.decision === 'approved' ? 'done' : 'failed'
          rt.summary = `${e.payload.decision} · ${e.payload.by}`
          rt.durationMs = rt.startedAt != null ? e.ts - rt.startedAt : null
        }

        break

      case 'LoopAdvanced':
        loopTarget = e.payload.to
        take = e.payload.iteration + 1

        break

      case 'FrameCommitted':

      case 'SnapshotCaptured':
        break
    }
  }

  // Edges follow from step state: a link is live while its target is queued
  // behind a finished source, and settled once the target has run.
  for (const def of shape.edges) {
    if (def.loop) {
      edges[def.id] = loopTarget === def.target ? 'loop' : 'idle'

      continue
    }

    const src = steps[def.source]
    const tgt = steps[def.target]
    const srcSettled = src.status === 'done' || src.status === 'failed'

    if (tgt.status === 'queued') {
      edges[def.id] = srcSettled ? 'active' : 'idle'
    } else if (tgt.status !== 'idle') {
      edges[def.id] = 'done'
    } else {
      edges[def.id] = 'idle'
    }
  }

  return { phase, take, steps, edges, clockTs }
}
