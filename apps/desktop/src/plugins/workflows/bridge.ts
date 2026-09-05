/**
 * The seam Hermes edits the canvas through.
 *
 * The `workflow` tool blocks in the Python agent and emits `workflow.request`;
 * this answers it. Everything it can do, it does by calling `callTool` — the
 * SAME dispatcher the inspector, the drag handles and the in-canvas composer
 * go through. There is no privileged path: an edit from a chat turn and an
 * edit from your hands are the same operation on the same document, land in
 * the same undo history, and are validated by the same rules.
 *
 * Two ways in, because the canvas may not be on screen:
 *
 *  - MOUNTED, and the tool is addressing the workflow that's open: the page
 *    has lent us its applier, so ops run against live React state and paint
 *    immediately, with undo and run control intact.
 *  - Otherwise: ops run against the stored scenario and are saved back. The
 *    tool still works with the user on another page — they just see the result
 *    when they come back.
 *
 * Model-facing contract lives in `GRAPH_TOOLS`, which `read` hands back
 * verbatim. That's deliberate: the op vocabulary is defined once, in
 * TypeScript, and the Python tool never restates it, so the two cannot drift.
 */

import { host } from '@hermes/plugin-sdk'

import { $currentId, $workflows, createWorkflow, saveWorkflow, type WorkflowDoc } from './documents'
import { fromScenario, type Graph, type OpResult, toScenario, validate } from './graph'
import { callTool, type RunControl } from './graph-dispatch'
import { GRAPH_TOOLS } from './graph-tools'
import { DEFAULT_DIR, type FlowDir } from './layout'
import { blankScenario } from './scenario'

export interface GraphOp {
  tool: string
  args?: Record<string, unknown>
}

/** What a mounted canvas lends the bridge: a way to run ops through the live
 *  editor rather than around it. Returns the graph as it ended up, so the
 *  reply can describe the real result and not the requested one. */
export interface LiveCanvas {
  id: string
  apply: (ops: GraphOp[]) => { graph: Graph; results: OpResult[] }
}

let live: LiveCanvas | null = null

/** Called by the canvas on mount. Returns the disposer it unmounts with. */
export function lendCanvas(canvas: LiveCanvas): () => void {
  live = canvas

  return () => {
    if (live === canvas) {
      live = null
    }
  }
}

// ---------------------------------------------------------------------------
// Shapes the model reads
// ---------------------------------------------------------------------------

/** A step list is what the model actually reasons over; the raw scenario is
 *  there for exactness. Both, because asking it to re-derive one from the
 *  other on every turn is how ids get invented. */
const describe = (doc: WorkflowDoc, graph: Graph) => ({
  workflow: { id: doc.id, name: doc.name },
  scenario: toScenario(graph),
  problems: validate(graph).map(p => ({ level: p.level, message: p.message, ...(p.step ? { step: p.step } : {}) })),
  // How to walk someone through the graph on screen.
  //
  // `selector` is stated once as a pattern rather than repeated per step: every
  // card carries the handle (see `cardProps` in nodes.tsx), so a step id from
  // the scenario above is already a tour target.
  //
  // The rules ride along because the `tour` tool is generic — it knows how to
  // point at a thing, and nothing about what a workflow is or who is being
  // shown one. Someone who asks for a tour is asking what this whole thing IS;
  // opening on a node card answers a question they haven't reached yet.
  tour: {
    rules: [
      'Open with a step that has NO selector. It centres on screen — that one is the whole idea of this workflow, in a sentence.',
      'Then follow the steps in run order, one per card.',
      'Explain it to someone who has never seen this app before, and never used one of these.',
      'One or two short sentences per step. Never a paragraph. No jargon.',
      'Say what a step DOES for the user, not how it is wired up.'
    ],
    selector: '[data-tour="step:<id>"]'
  }
})

const summarise = (doc: WorkflowDoc) => ({
  id: doc.id,
  name: doc.name,
  steps: doc.scenario.steps.length
})

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

const openDoc = (): undefined | WorkflowDoc => $workflows.get().find(d => d.id === $currentId.get())

/** Resolve a workflow the way a person would name one: by id, or by name,
 *  case-insensitively. */
const findDoc = (ref: string): undefined | WorkflowDoc => {
  const needle = ref.trim().toLowerCase()

  return $workflows.get().find(d => d.id.toLowerCase() === needle || d.name.toLowerCase() === needle)
}

/** Bring the canvas to the front. `open` and `create` are the two verbs whose
 *  whole point is which workflow you're looking at, and answering them without
 *  showing it leaves the user on some other page being told about a switch
 *  they can't see. It also puts the nodes in the DOM, which is what makes the
 *  `tour` tool able to point at them. */
const show = () => host.navigate('/workflows')

const NOTHING_OPEN = {
  error:
    'No workflow is open. Call action="list" to see what there is, action="open" to switch to one, ' +
    'or action="create" to start a new one.'
}

/** Run a batch, folding each op's graph into the next. Refusals don't stop the
 *  batch: an op the schema rejects is usually one bad argument in an otherwise
 *  good plan, and halting would leave the graph half-edited with no way for
 *  the model to tell which half. It gets told exactly what didn't land. */
export function runOps(
  graph: Graph,
  run: RunControl,
  ops: GraphOp[],
  dir: FlowDir = DEFAULT_DIR
): { graph: Graph; results: OpResult[] } {
  let next = graph
  const results: OpResult[] = []

  for (const op of ops) {
    const result = callTool(next, run, op.tool, op.args ?? {}, dir)
    results.push(result)

    if (result.ok) {
      next = result.graph
    }
  }

  return { graph: next, results }
}

/** Run control needs a player, and the player belongs to the mounted canvas.
 *  Off screen there is nothing to drive, so say so rather than silently
 *  reporting a run that never started. */
const NO_RUN: RunControl = {
  running: false,
  paused: false,
  start: () => {
    throw new Error('Open the Workflows page to run a workflow.')
  },
  pause: () => {
    throw new Error('Nothing is running.')
  },
  resume: () => {
    throw new Error('Nothing is running.')
  },
  reset: () => {
    throw new Error('Nothing is running.')
  }
}

function edit(ops: GraphOp[]): Record<string, unknown> {
  const doc = openDoc()

  if (!doc) {
    return NOTHING_OPEN
  }

  // The mounted canvas owns the authoritative graph while it's on screen —
  // it holds positions and selection the stored scenario only learns about on
  // save — so ops have to go through it rather than around it.
  const applied =
    live?.id === doc.id
      ? live.apply(ops)
      : (() => {
          const result = runOps(fromScenario(doc.scenario), NO_RUN, ops)
          saveWorkflow(doc.id, toScenario(result.graph))

          return result
        })()

  const { graph, results } = applied

  return {
    ...describe(doc, graph),
    applied: results.filter(r => r.ok).map(r => r.message),
    refused: results.map((r, i) => ({ op: ops[i].tool, why: r.message })).filter((_, i) => !results[i].ok)
  }
}

function act(payload: Record<string, unknown>): Record<string, unknown> {
  const action = typeof payload.action === 'string' ? payload.action : ''
  const ref = typeof payload.workflow === 'string' ? payload.workflow : ''

  switch (action) {
    case 'read': {
      const doc = openDoc()

      // The op vocabulary rides along with the read, so the model is holding
      // the live schema when it composes the edit rather than whatever the
      // Python description last claimed.
      return doc
        ? { ...describe(doc, fromScenario(doc.scenario)), ops: GRAPH_TOOLS }
        : { ...NOTHING_OPEN, ops: GRAPH_TOOLS }
    }

    case 'edit':
      return edit(Array.isArray(payload.ops) ? (payload.ops as GraphOp[]) : [])

    case 'list':
      return { workflows: $workflows.get().map(summarise), open: $currentId.get() }
    case 'open': {
      const doc = findDoc(ref)

      if (!doc) {
        return {
          error: `There's no workflow called "${ref}".`,
          workflows: $workflows.get().map(summarise)
        }
      }

      $currentId.set(doc.id)
      show()

      return { opened: summarise(doc), ...describe(doc, fromScenario(doc.scenario)) }
    }

    case 'create': {
      const scenario = (payload.scenario as WorkflowDoc['scenario'] | undefined) ?? blankScenario()
      const id = createWorkflow(ref, scenario)
      const doc = $workflows.get().find(d => d.id === id)

      show()

      return doc
        ? { created: summarise(doc), ...describe(doc, fromScenario(doc.scenario)) }
        : { error: 'Could not create it.' }
    }

    default:
      return { error: `Unknown action "${action}". Use read, edit, list, open or create.` }
  }
}

// ---------------------------------------------------------------------------
// The wire
// ---------------------------------------------------------------------------

/** Listen for the tool's blocking request and answer it. Returns the disposer,
 *  for `ctx.onDispose`. */
export function bindBridge(): () => void {
  return host.onEvent('workflow.request', event => {
    const payload = (event.payload ?? {}) as Record<string, unknown>
    const requestId = typeof payload.request_id === 'string' ? payload.request_id : ''

    if (!requestId) {
      return
    }

    // A throw here would strand the agent on the full 30s timeout, so every
    // exit from this handler answers — including the failure.
    let answer: Record<string, unknown>

    try {
      answer = act(payload)
    } catch (error) {
      answer = { error: error instanceof Error ? error.message : String(error) }
    }

    void host.request('workflow.respond', { request_id: requestId, text: JSON.stringify(answer) })
  })
}
