// The wire boundary between the canvas and whatever runs the scenario.
//
// Shaped on purpose like the Smithers gateway's run-event log
// (`_smithers_events`: run_id, seq, timestamp_ms, type, payload_json), so a
// real Smithers backend can drive this UI through a thin adapter instead of a
// rewrite. Checked against a live gateway run of our own north star
// (run-1784922888902): every SMITHERS_NATIVE type below is one it actually
// emits, keyed the same way — (runId, stepId, take).
//
// NAMING: the event type names are the ENGINE's and are deliberately left
// verbatim — `NodeStarted`, `FrameCommitted`, `iteration`. They're a contract
// with a system we don't own. Everything the *canvas* says is in the canvas's
// own lexicon: scenario, run, step, take, checkpoint, split, replay. Where the
// two meet, the adapter translates and the field comment says so.
//
// The HERMES_EXT types are the gap. Smithers keeps join/branch decisions in JS
// closures, so its wire never sees them: that run produced zero gate steps and
// zero edges. Those events are the canvas's own, and they're the whole cost of
// the translation — small, and on our side of the line.
//
// Nothing here knows how the run is produced. The gateway runner is one
// adapter; a scripted fixture would be another.
//
// This file is the vocabulary. What you can DERIVE from a stream of it lives
// next door: `protocol-world.ts` folds events into the world the canvas
// renders, `protocol-feed.ts` projects the same events as readable lines.

import type { OnFail } from './scenario'

/** The shape of the thing being run — which steps exist and how they're wired.
 *
 *  The reducer used to read this off the static STEP_DEFS/EDGE_DEFS, which
 *  meant the world had exactly the six steps the starter scenario ships with.
 *  Every event
 *  about a step you added landed on `steps[id] === undefined` and was dropped,
 *  so a node you drew could never light up no matter what the engine said
 *  about it. The canvas has never seen an engine; it hadn't seen the graph
 *  either. */
export interface RunShape {
  steps: string[]
  edges: { id: string; source: string; target: string; loop?: boolean }[]
}

// ---------------------------------------------------------------------------
// Payload pieces
// ---------------------------------------------------------------------------

/** A single tool call the subagent makes (Hermes `get_activity_summary()`). */
export interface ToolCall {
  name: string
  arg: string
}

/** Mirror of the todo tool's item shape. */
export type TodoStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled'
export interface TodoItem {
  id: string
  content: string
  status: TodoStatus
}

export type Verdict = 'PASS' | 'FAIL' | null

/** Every step event is addressed the way Smithers addresses it. */
export interface NodeRef {
  nodeId: string
  /** The engine's word for a take. Wire-level name, kept verbatim. */
  iteration: number
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

export type EventType =
  // --- Smithers emits these today, verbatim --------------------------------
  | 'RunStarted'
  | 'RunFinished'
  | 'NodePending'
  | 'NodeStarted'
  | 'NodeFinished'
  | 'NodeFailed'
  | 'AgentTraceEvent'
  | 'AgentTraceSummary'
  | 'TaskOutput'
  | 'TokenUsage'
  | 'FrameCommitted'
  | 'SnapshotCaptured'
  // --- canvas extensions: no Smithers equivalent on the wire ---------------
  | 'GateEvaluated'
  | 'NodeSkipped'
  | 'LoopAdvanced'
  | 'TodoUpdated'
  | 'HumanWaiting'
  | 'HumanResponded'
  | 'WaitStarted'
  | 'WaitResolved'
  | 'RunPaused'
  | 'UserAsk'

/** Types the canvas has to synthesize because the engine doesn't report them. */
export const HERMES_EXT: ReadonlySet<EventType> = new Set<EventType>([
  'GateEvaluated',
  'NodeSkipped',
  'LoopAdvanced',
  'TodoUpdated',
  // Smithers wire-wise a HumanTask is just a task that stays running; the
  // park/resume pair is the canvas's own so a card can say "waiting on you".
  'HumanWaiting',
  'HumanResponded',
  // The same pair for the other thing a run parks on. A wait and a human step
  // both stop the run dead, but "waiting on you" is wrong over a timer, so the
  // world gets its own two events rather than borrowing the person's.
  'WaitStarted',
  'WaitResolved',
  'RunPaused',
  'UserAsk'
])

interface Envelope {
  runId: string
  seq: number
  ts: number
}

export type ProtoEvent =
  | (Envelope & { type: 'RunStarted'; payload: { scenario: string } })
  | (Envelope & { type: 'RunFinished'; payload: { state: 'succeeded' | 'failed' } })
  | (Envelope & { type: 'RunPaused'; payload: Record<string, never> })
  | (Envelope & { type: 'NodePending'; payload: NodeRef })
  | (Envelope & {
      type: 'NodeStarted'
      payload: NodeRef & { input: string; maxIters: number; loop?: boolean }
    })
  | (Envelope & { type: 'NodeFinished'; payload: NodeRef })
  | (Envelope & { type: 'NodeFailed'; payload: NodeRef & { error: string } })
  | (Envelope & {
      type: 'AgentTraceEvent'
      payload: NodeRef & { tool: ToolCall }
    })
  | (Envelope & {
      type: 'AgentTraceSummary'
      payload: NodeRef & { summary: string; verdict?: Verdict }
    })
  | (Envelope & {
      type: 'TaskOutput'
      payload: NodeRef & { output: Record<string, unknown> }
    })
  | (Envelope & { type: 'TokenUsage'; payload: NodeRef & { tokens: number } })
  | (Envelope & { type: 'FrameCommitted'; payload: { frameNo: number; label: string } })
  | (Envelope & { type: 'SnapshotCaptured'; payload: { frameNo: number } })
  | (Envelope & {
      type: 'GateEvaluated'
      payload: NodeRef & {
        inputs: { nodeId: string; verdict: Verdict }[]
        decision: 'pass' | 'fail'
        route: string
        summary: string
      }
    })
  | (Envelope & { type: 'NodeSkipped'; payload: NodeRef & { reason: string } })
  | (Envelope & {
      type: 'HumanWaiting'
      payload: NodeRef & {
        prompt: string
        /** Who it's parked on, for the card and the prompt. */
        who: string
        /** What a "no" means here, so the answer can be honoured rather than
         *  assumed. It's the step's own on-failure setting: a denial IS the
         *  failure. */
        onFail: OnFail
      }
    })
  | (Envelope & {
      type: 'HumanResponded'
      payload: NodeRef & { decision: 'approved' | 'denied'; by: string }
    })
  | (Envelope & {
      type: 'WaitStarted'
      payload: NodeRef & { until: string; label: string }
    })
  | (Envelope & { type: 'WaitResolved'; payload: NodeRef & { by: string } })
  | (Envelope & {
      type: 'LoopAdvanced'
      payload: { loopId: string; iteration: number; to: string; feedback: string }
    })
  | (Envelope & { type: 'TodoUpdated'; payload: NodeRef & { todos: TodoItem[] } })
  | (Envelope & { type: 'UserAsk'; payload: NodeRef & { prompt: string } })

// ---------------------------------------------------------------------------
// Derived state — everything the canvas draws is a fold over the event stream.
// ---------------------------------------------------------------------------

export type StepStatus =
  | 'idle'
  | 'queued'
  | 'running'
  | 'waiting' // parked on a person (or the world) — the run is not working
  | 'done'
  | 'failed'
  | 'looping'

export interface StepRuntime {
  status: StepStatus
  currentTool: ToolCall | null
  toolCalls: ToolCall[]
  todos: TodoItem[]
  iterations: number
  maxIters: number
  tokens: number
  /** Which take this is — 1-based. A gate rejection sends the step back for
   *  another one, the way a director calls take 2. */
  take: number
  startedAt: number | null
  durationMs: number | null
  verdict: Verdict
  input: string | null
  summary: string | null
  output: Record<string, unknown> | null
  skipped: string | null
}

export type EdgeState = 'idle' | 'active' | 'done' | 'loop'
export type RunPhase = 'idle' | 'running' | 'done'

/** A URL a step produced, ready to surface on the card. */
export interface StepLink {
  /** The output key it came from, e.g. `pr_url`. */
  key: string
  href: string
  /** Short human label — the origin + last path segment, e.g. `github.com/…/1234`. */
  label: string
}

/**
 * Find the first URL a step emitted, so the canvas can link to it directly.
 *
 * A run's whole point is usually the artifact at the end — a PR, a preview, a
 * dashboard. Making that reachable only through select → Data tab → read the
 * value buries the one thing you came for behind three clicks. This scans the
 * structured output generically rather than special-casing `pr_url`, so any
 * step that returns a URL gets a link for free.
 */
export function stepLink(output: Record<string, unknown> | null): StepLink | null {
  if (!output) {
    return null
  }

  for (const [key, value] of Object.entries(output)) {
    if (typeof value !== 'string') {
      continue
    }

    if (!/^https?:\/\//i.test(value)) {
      continue
    }

    try {
      const u = new URL(value)
      const tail = u.pathname.split('/').filter(Boolean).pop()

      return {
        key,
        href: value,
        label: tail ? `${u.host}/…/${tail}` : u.host
      }
    } catch {
      // Malformed URL — skip it rather than rendering a dead link.
    }
  }

  return null
}

/**
 * A durable boundary you can scrub to — the engine's `FrameCommitted`.
 *
 * "Checkpoint", not "frame": a frame is a ~16ms display tick to anyone with
 * emulator, TAS, or video literacy, and this is a semantic save point, not a
 * time quantum. Checkpoint already means "the place you resume from" in both
 * games and software, so it needs no explaining in either direction.
 */
export interface Checkpoint {
  no: number
  label: string
  /** Index into the event array, so seeking a checkpoint is seeking an event. */
  at: number
}

export interface World {
  phase: RunPhase
  /** Highest take reached by any step in the run so far. */
  take: number
  steps: Record<string, StepRuntime>
  edges: Record<string, EdgeState>
  /** Timestamp of the last applied event — the clock while time-travelling. */
  clockTs: number | null
}

export const freshRuntime = (): StepRuntime => ({
  status: 'idle',
  currentTool: null,
  toolCalls: [],
  todos: [],
  iterations: 0,
  maxIters: 0,
  tokens: 0,
  take: 0,
  startedAt: null,
  durationMs: null,
  verdict: null,
  input: null,
  summary: null,
  output: null,
  skipped: null
})

/** The agent returned an exception string as if it were a summary. */
export function isBrokenReply(text: string | null | undefined): boolean {
  const raw = (text ?? '').trim().replace(/^["']|["']$/g, '')

  if (!raw) {
    return false
  }

  const low = raw.toLowerCase()

  return (
    raw.startsWith('HTTP 4') ||
    raw.startsWith('HTTP 5') ||
    low.includes('could not resolve authentication') ||
    low.includes('api_key or auth_token') ||
    low.includes('model parameter is required') ||
    low.includes('requested model does not exist') ||
    low.includes('could not load the agent')
  )
}
