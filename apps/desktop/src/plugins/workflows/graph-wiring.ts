// Wires: drawing one, cutting one, and making sure every wire out of a gate
// leaves by an arm that exists.

import { armsOf, freeArmId, guessWhen, withArms } from './graph-arms'
import {
  dataOf,
  edgeIdFor,
  fail,
  type Graph,
  isLoop,
  newEdge,
  type OpResult,
  reaches,
  resolveStep,
  stepNodes
} from './graph-core'
import { type Arm, NEW_BRANCH, type Predicate } from './scenario'

/** Give every wire leaving a gate an arm to leave by.
 *
 *  `connect` mints one as it goes, but the wires addStep lays down — splicing
 *  onto an edge, or hanging a step off an `after` — are built directly, and a
 *  gate wire with no handle is one the card can't draw and the run can't
 *  follow. Splice a gate into the middle of a flow and the work stopped there
 *  silently: the gate picked an arm, the arm named no wire, and everything
 *  downstream just never ran. */
export function armWires(g: Graph): Graph {
  const gates = new Set(
    stepNodes(g)
      .filter(n => dataOf(n).def.kind === 'gate')
      .map(n => n.id)
  )

  if (!gates.size) {
    return g
  }

  let nodes = g.nodes
  const taken = new Set(g.edges.filter(e => e.sourceHandle).map(e => `${e.source}/${e.sourceHandle}`))

  const edges = g.edges.map(e => {
    if (!gates.has(e.source)) {
      return e
    }

    const arms = armsOf({ nodes, edges: g.edges }, e.source)
    const named = e.sourceHandle && e.sourceHandle !== NEW_BRANCH

    if (named && arms.some(a => a.id === e.sourceHandle)) {
      return e
    }

    // The spare arm if there is one, a fresh one if there isn't — same choice
    // connect makes, so a wire laid down here is indistinguishable from a drawn
    // one.
    const spare = arms.find(a => !taken.has(`${e.source}/${a.id}`))

    const arm: Arm = spare ?? {
      id: freeArmId(
        arms.map(a => a.id),
        isLoop(e) ? 'loop' : 'pass'
      ),
      when: guessWhen(arms, isLoop(e))
    }

    if (!spare) {
      nodes = withArms({ nodes, edges: g.edges }, e.source, [...arms, arm])
    }

    taken.add(`${e.source}/${arm.id}`)

    return { ...e, sourceHandle: arm.id }
  })

  return { nodes, edges }
}

export interface ConnectInput {
  source: string
  target: string
  when?: Predicate
  sourceHandle?: string
  targetHandle?: string
}

export function connect(g: Graph, input: ConnectInput): OpResult {
  const from = resolveStep(g, input.source)
  const to = resolveStep(g, input.target)

  if (!from) {
    return fail(g, `There's no step called "${input.source}".`)
  }

  if (!to) {
    return fail(g, `There's no step called "${input.target}".`)
  }

  if (from.id === to.id) {
    return fail(g, "A step can't feed itself.")
  }

  if (g.edges.some(e => e.source === from.id && e.target === to.id)) {
    return fail(g, `${from.id} already feeds ${to.id}.`)
  }

  // A wire that closes a cycle is a rework loop, and the canvas draws those
  // differently (deep belly under the flow, amber, held out of Dagre so the
  // graph still has a rank order to lay out). Detect it here rather than
  // asking the author to classify their own edge.
  const loop = reaches(g, to.id, from.id)
  const gate = dataOf(from)?.def?.kind === 'gate'

  // Out of a gate the wire has to leave by an arm. Naming one that already
  // exists claims it; the spare port (or a tool with no opinion) mints a new
  // one. Two wires can share an arm — that's a fan-out on one condition — so
  // claiming isn't exclusive.
  let nodes = g.nodes
  let sourceHandle = input.sourceHandle

  if (gate) {
    const arms = armsOf(g, from.id)
    const asked = input.sourceHandle
    const claimed = asked && asked !== NEW_BRANCH ? arms.find(a => a.id === asked) : undefined

    if (claimed) {
      sourceHandle = claimed.id

      if (input.when) {
        nodes = withArms(
          g,
          from.id,
          arms.map(a => (a.id === claimed.id ? { ...a, when: input.when! } : a))
        )
      }
    } else {
      const arm: Arm = {
        id: freeArmId(
          arms.map(a => a.id),
          loop ? 'loop' : 'pass'
        ),
        when: input.when ?? guessWhen(arms, loop)
      }

      sourceHandle = arm.id
      nodes = withArms(g, from.id, [...arms, arm])
    }
  }

  const edge = newEdge(from.id, to.id, {
    loop,
    sourceHandle,
    targetHandle: input.targetHandle ?? (loop ? 'loopback' : undefined)
  })

  return {
    ok: true,
    graph: { nodes, edges: [...g.edges, edge] },
    message: loop ? `Wired ${from.id} back to ${to.id} as a rework loop.` : `Wired ${from.id} into ${to.id}.`,
    edit: `${from.id} → ${to.id}`
  }
}

export function disconnect(g: Graph, source: string, target?: string): OpResult {
  const edge = target
    ? g.edges.find(e => (e.source === source && e.target === target) || e.id === edgeIdFor(source, target))
    : g.edges.find(e => e.id === source)

  if (!edge) {
    return fail(g, "There's no wire like that to cut.")
  }

  return {
    ok: true,
    graph: { nodes: g.nodes, edges: g.edges.filter(e => e.id !== edge.id) },
    message: `Cut ${edge.source} → ${edge.target}.`,
    edit: `− ${edge.source} → ${edge.target}`
  }
}
