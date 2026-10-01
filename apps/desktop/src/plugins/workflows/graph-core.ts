// The document itself, and the handful of words every op needs to talk about
// it: what a graph is, what an op reports back, how a step is named and found,
// and how a wire is built and followed.
//
// Nothing here changes the document. The ops live in the modules beside this
// one and all of them start from these.

import type { Edge, Node } from '@xyflow/react'

import type { NodeData } from './nodes'

export interface Graph {
  nodes: Node[]
  edges: Edge[]
}

/** What every op reports back: the next graph, plus a line fit for the composer
 *  transcript and the applied-edit chip. `ok: false` leaves the graph alone —
 *  a refused edit is a normal outcome, not an exception. */
export interface OpResult {
  ok: boolean
  graph: Graph
  message: string
  /** Short mutation summary, e.g. `+ step lint → gate`. Absent when nothing moved. */
  edit?: string
  /** Set by ops that mint something, so a caller can select it. */
  focus?: string
}

export const fail = (graph: Graph, message: string): OpResult => ({ ok: false, graph, message })

export const dataOf = (n: Node) => n.data as unknown as NodeData

export const stepById = (g: Graph, id: string) => g.nodes.find(n => n.id === id)

export const stepNodes = (g: Graph) => g.nodes.filter(n => !!dataOf(n)?.def)

/** Resolve a step the way a person would name it: by id first, then by title,
 *  case-insensitively. An agent that says "the visual judge" should not have to
 *  know we called it `judge`. */
export function resolveStep(g: Graph, ref: string): Node | undefined {
  const needle = ref.trim().toLowerCase()
  const steps = stepNodes(g)

  return (
    steps.find(n => n.id.toLowerCase() === needle) ??
    steps.find(n => dataOf(n).config.title.toLowerCase() === needle) ??
    steps.find(n => dataOf(n).config.title.toLowerCase().includes(needle))
  )
}

/** Stable, free, and readable. Ids are what gate rules and `needs:` refer to,
 *  so a minted one is derived from the title rather than a counter — `step-3`
 *  tells a later reader nothing, and an agent authoring a graph would have to
 *  invent names for its own conditions anyway. */
export function mintId(g: Graph, from: string): string {
  const base =
    from
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 24) || 'step'

  if (!stepById(g, base)) {
    return base
  }

  for (let i = 2; ; i++) {
    if (!stepById(g, `${base}_${i}`)) {
      return `${base}_${i}`
    }
  }
}

export const edgeIdFor = (source: string, target: string) => `${source}->${target}`

export function newEdge(
  source: string,
  target: string,
  opts: { sourceHandle?: string; targetHandle?: string; loop?: boolean } = {}
): Edge {
  return {
    id: edgeIdFor(source, target),
    source,
    target,
    ...(opts.sourceHandle ? { sourceHandle: opts.sourceHandle } : {}),
    ...(opts.targetHandle ? { targetHandle: opts.targetHandle } : {}),
    type: 'data',
    data: { state: 'idle', loop: opts.loop }
  }
}

export const isLoop = (e: Edge) => Boolean((e.data as { loop?: boolean })?.loop)

/** Can `from` reach `to` following forward wires? Used to tell a rework loop
 *  from an ordinary wire, and to find steps the run can never arrive at. */
export function reaches(g: Graph, from: string, to: string): boolean {
  const seen = new Set<string>()
  const stack = [from]

  while (stack.length) {
    const at = stack.pop()!

    if (at === to) {
      return true
    }

    if (seen.has(at)) {
      continue
    }

    seen.add(at)

    for (const e of g.edges) {
      if (e.source === at && !isLoop(e)) {
        stack.push(e.target)
      }
    }
  }

  return false
}
