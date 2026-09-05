// The cards: adding one, dropping one, editing one, renaming it, and turning
// it into a different kind of step.

import type { Edge, Node, XYPosition } from '@xyflow/react'

import { freeArmId, guessWhen } from './graph-arms'
import { dataOf, edgeIdFor, fail, type Graph, isLoop, mintId, newEdge, type OpResult, resolveStep, stepById } from './graph-core'
import { armWires } from './graph-wiring'
import { DEFAULT_DIR, type FlowDir, freeRow, freeSpot, heightOf, RANK_GAP, widthOf } from './layout'
import type { NodeData } from './nodes'
import { freshRuntime } from './protocol'
import {
  type Arm,
  defaultConfig,
  FIELD_LABEL,
  type KindField,
  type Predicate,
  pruneConfig,
  STEP_KINDS,
  type StepConfig,
  type StepDef,
  type StepKind
} from './scenario'

export interface AddStepInput {
  kind: StepKind
  title?: string
  goal?: string
  /** Splice into this wire: the new step takes its place in the middle. */
  onEdge?: string
  /** Wire straight out of this step. */
  after?: string
  /** Wire straight into this step. */
  before?: string
  position?: XYPosition
  config?: Partial<StepConfig>
  /** Which way the ranks run, for placing a wired step one rank along. The
   *  canvas passes its current direction; a headless edit takes the default. */
  dir?: FlowDir
}

export function addStep(g: Graph, input: AddStepInput): OpResult {
  const spec = STEP_KINDS.find(k => k.kind === input.kind)

  if (!spec) {
    return fail(g, `There's no "${input.kind}" kind — use agent, gate, human, wait or trigger.`)
  }

  const title = input.title?.trim() || spec.title
  const id = mintId(g, title)
  const def: StepDef = { id, kind: input.kind, title, doing: spec.doing }

  const config: StepConfig = {
    ...defaultConfig(def),
    ...(input.goal !== undefined ? { goal: input.goal } : {}),
    ...input.config,
    title
  }

  let edges = g.edges
  let wiring = ''
  // A tool almost never sends a position — an agent has no business knowing
  // the canvas's geometry. So the step is placed next to whatever it was
  // wired to, one rank along in the direction the flow runs, and nudged clear
  // of anything already sitting there. Nothing the author placed by hand moves.
  let anchor: Node | undefined
  let lead = 1

  if (input.onEdge) {
    const split = g.edges.find(e => e.id === input.onEdge)

    if (!split) {
      return fail(g, `There's no wire called "${input.onEdge}".`)
    }

    edges = [
      ...g.edges.filter(e => e.id !== split.id),
      newEdge(split.source, id, { sourceHandle: split.sourceHandle ?? undefined }),
      newEdge(id, split.target, { targetHandle: split.targetHandle ?? undefined })
    ]
    wiring = ` between ${split.source} and ${split.target}`
    anchor = stepById(g, split.source)
  } else {
    const links: Edge[] = []

    if (input.after) {
      const from = resolveStep(g, input.after)

      if (!from) {
        return fail(g, `There's no step called "${input.after}".`)
      }

      links.push(newEdge(from.id, id))
      anchor = from
    }

    if (input.before) {
      const to = resolveStep(g, input.before)

      if (!to) {
        return fail(g, `There's no step called "${input.before}".`)
      }

      links.push(newEdge(id, to.id))

      if (!anchor) {
        anchor = to
        lead = -1
      }
    }

    edges = [...g.edges, ...links]

    if (links.length) {
      wiring = ` ${input.after ? `after ${input.after}` : ''}${
        input.after && input.before ? ' and' : ''
      }${input.before ? ` before ${input.before}` : ''}`
    }
  }

  // A tool never sends a position, so the rank step has to follow the flow.
  // It used to be hardcoded along x, which read correctly only while the
  // canvas was horizontal — once vertical became the default, every step an
  // agent wired on marched SIDEWAYS while its wire ran downward, and a graph
  // built from scratch came out as a row of cards stitched together by
  // zigzags.
  const dir = input.dir ?? DEFAULT_DIR

  /** One rank along from `from`, in the direction the flow runs. */
  const nextRank = (from: Node) =>
    dir === 'TB'
      ? { x: from.position.x, y: from.position.y + lead * (heightOf(from) + RANK_GAP) }
      : { x: from.position.x + lead * (widthOf(from) + RANK_GAP), y: from.position.y }

  const position =
    input.position ?? (anchor ? freeRow(g.nodes, nextRank(anchor), dir) : freeSpot(g.nodes, { x: 0, y: 0 }, dir))

  const node: Node = {
    id,
    type: input.kind,
    position,
    data: { def, config, rt: freshRuntime(), selected: false } satisfies NodeData
  }

  return {
    ok: true,
    graph: armWires({ nodes: [...g.nodes, node], edges }),
    message: `Added ${title}${wiring}.`,
    edit: `+ step ${id}`,
    focus: id
  }
}

/** n8n's delete: a card with one hop in and one hop out reconnects its
 *  neighbours, so A → X → B becomes A → B with the ports preserved. Anything
 *  branching just loses its wires — healing a 2-in/2-out gate into four new
 *  edges would invent a topology nobody drew. */
export function removeStep(g: Graph, ref: string): OpResult {
  const node = resolveStep(g, ref)

  if (!node) {
    return fail(g, `There's no step called "${ref}".`)
  }

  const ins = g.edges.filter(e => e.target === node.id && !isLoop(e))
  const outs = g.edges.filter(e => e.source === node.id && !isLoop(e))
  const kept = g.edges.filter(e => e.source !== node.id && e.target !== node.id)

  const heal =
    ins.length === 1 &&
    outs.length === 1 &&
    ins[0].source !== outs[0].target &&
    !kept.some(e => e.source === ins[0].source && e.target === outs[0].target)
      ? [
          newEdge(ins[0].source, outs[0].target, {
            sourceHandle: ins[0].sourceHandle ?? undefined,
            targetHandle: outs[0].targetHandle ?? undefined
          })
        ]
      : []

  return {
    ok: true,
    graph: {
      nodes: g.nodes.filter(n => n.id !== node.id),
      edges: [...kept, ...heal]
    },
    message: `Removed ${dataOf(node).config.title}.`,
    edit: `− step ${node.id}`
  }
}

export function updateStep(g: Graph, ref: string, patch: Partial<StepConfig>): OpResult {
  const node = resolveStep(g, ref)

  if (!node) {
    return fail(g, `There's no step called "${ref}".`)
  }

  if (!Object.keys(patch).length) {
    return fail(g, 'Nothing to change.')
  }

  // A patch is cut to the kind before it lands. Saying so matters more than
  // silently dropping it: "set the model on this timer" is a misunderstanding
  // worth answering, and an agent that gets told which knob doesn't exist can
  // pick the step it actually meant.
  const { kind } = dataOf(node).def
  const keys = Object.keys(pruneConfig(kind, patch))
  const refused = Object.keys(patch).filter(k => !keys.includes(k))

  if (!keys.length) {
    return fail(g, `${kind} steps have no ${listOf(refused)}.`)
  }

  return {
    ok: true,
    graph: {
      nodes: g.nodes.map(n =>
        n.id === node.id
          ? {
              ...n,
              data: { ...n.data, config: { ...dataOf(n).config, ...pruneConfig(kind, patch) } }
            }
          : n
      ),
      edges: g.edges
    },
    message: refused.length
      ? `Updated ${dataOf(node).config.title} — ${kind} steps have no ${listOf(refused)}.`
      : `Updated ${dataOf(node).config.title}.`,
    edit: `${node.id} · ${keys.join(', ')}`,
    focus: node.id
  }
}

/** "a", "a and b", "a, b and c" — for naming the knobs a kind doesn't have, in
 *  the words the panel uses rather than the wire names. */
function listOf(keys: string[]): string {
  const xs = keys.map(k => FIELD_LABEL[k as KindField] ?? k)

  if (xs.length < 2) {
    return xs[0] ?? ''
  }

  return `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`
}

/** Renaming the id rewrites every reference to it — wires, handles and any gate
 *  rule that names it. The id is the only handle the rest of the scenario has
 *  on a step, so a rename that missed one would quietly break routing. */
export function renameStep(g: Graph, ref: string, nextId: string): OpResult {
  const node = resolveStep(g, ref)

  if (!node) {
    return fail(g, `There's no step called "${ref}".`)
  }

  const clean = nextId
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')

  if (!clean) {
    return fail(g, 'An id needs at least one letter or number.')
  }

  if (clean === node.id) {
    return fail(g, "That's already its id.")
  }

  if (stepById(g, clean)) {
    return fail(g, `"${clean}" is already taken.`)
  }

  const swap = (id: string) => (id === node.id ? clean : id)

  return {
    ok: true,
    graph: {
      nodes: g.nodes.map(n => {
        const arms = (n.data as NodeData).config.arms

        const renamed = arms?.map(a => ({
          ...a,
          when: renamePredicate(a.when, node.id, clean) ?? a.when
        }))

        const data = {
          ...n.data,
          ...(renamed ? { config: { ...(n.data as NodeData).config, arms: renamed } } : {}),
          ...(n.id === node.id ? { def: { ...dataOf(n).def, id: clean } } : {})
        }

        return n.id === node.id ? { ...n, id: clean, data } : { ...n, data }
      }),
      edges: g.edges.map(e => {
        if (e.source !== node.id && e.target !== node.id) {
          return e
        }

        const source = swap(e.source)
        const target = swap(e.target)

        return { ...e, id: edgeIdFor(source, target), source, target }
      })
    },
    message: `Renamed ${node.id} to ${clean}.`,
    edit: `${node.id} → ${clean}`,
    focus: clean
  }
}

const renamePredicate = (p: Predicate | undefined, from: string, to: string): Predicate | undefined =>
  p?.mode === 'checks' ? { ...p, checks: p.checks.map(c => (c.step === from ? { ...c, step: to } : c)) } : p

/** Turn a step into a different kind of step.
 *
 *  The kind decides which config fields mean anything and how many outputs the
 *  card has, so this rebuilds the config from the kind's defaults and keeps
 *  only what survives the change — the name, the instruction, and the wiring.
 *
 *  The wiring is the part that can't be skipped. A gate names each arm's port
 *  after the branch it is ("pass", "loop_2"); every other kind has exactly one
 *  output, called "out". Leave those handles alone and the wires point at
 *  ports the card no longer renders, which reads as edges flying to a corner. */
export function setKind(g: Graph, ref: string, kind: StepKind): OpResult {
  const node = resolveStep(g, ref)

  if (!node) {
    return fail(g, `There's no step called "${ref}".`)
  }

  const spec = STEP_KINDS.find(k => k.kind === kind)

  if (!spec) {
    return fail(g, `There's no "${kind}" kind — use agent, gate, human, wait or trigger.`)
  }

  const data = dataOf(node)
  const was = data.def.kind

  if (was === kind) {
    return fail(g, `${node.id} is already a ${kind}.`)
  }

  const def: StepDef = { ...data.def, kind, doing: spec.doing }

  // Whatever the two kinds share comes across — the name always, the
  // instruction and the timeout between an agent and a human — and the new
  // kind's defaults fill what it gained. Everything else is dropped by the
  // prune rather than by a list here, so a step converted twice can't arrive
  // back carrying a knob it lost on the way out.
  const config: StepConfig = {
    ...defaultConfig(def),
    ...pruneConfig(kind, data.config)
  } as StepConfig

  // Becoming a gate gives every wire already leaving an arm to leave by;
  // ceasing to be one collapses them all onto the single output every other
  // kind has. Skip either and the wires point at ports the card no longer
  // renders, which reads as edges flying off to a corner.
  let edges = g.edges

  if (kind === 'gate') {
    // The wires already leaving decide the table — one arm each, replacing the
    // pass/fail pair a gate is born with, which is only right for a gate made
    // from nothing. Keeping both is what duplicated the ids.
    const out = g.edges.filter(e => e.source === node.id)

    if (out.length) {
      const taken: string[] = []
      const arms: Arm[] = []

      for (const e of out) {
        const loop = isLoop(e)

        const arm: Arm = {
          id: freeArmId(taken, loop ? 'loop' : 'pass'),
          when: guessWhen(arms, loop)
        }

        arms.push(arm)
        edges = edges.map(x => (x.id === e.id ? { ...x, sourceHandle: arm.id } : x))
      }

      config.arms = arms
    }
  } else if (was === 'gate') {
    edges = g.edges.map(e => (e.source === node.id ? { ...e, sourceHandle: 'out' } : e))
  }

  return {
    ok: true,
    graph: {
      nodes: g.nodes.map(n => (n.id === node.id ? { ...n, type: kind, data: { ...n.data, def, config } } : n)),
      edges
    },
    message: `${node.id} is a ${kind} step now.`,
    edit: `${node.id} · ${was} → ${kind}`
  }
}
