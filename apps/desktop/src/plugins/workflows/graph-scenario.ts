// The boundary between the canvas's document and the authored scenario.
//
// Everything downstream of the canvas — the gateway runner, the tool payloads,
// what lands on disk — speaks `Scenario`, so React Flow's node and edge shapes
// stop here. `runPlan` is the same boundary for a run: the executor takes the
// plan rather than the graph so it never has to know what a handle is.

import type { Edge, Node } from '@xyflow/react'

import { freeArmId, guessWhen } from './graph-arms'
import { dataOf, edgeIdFor, type Graph, isLoop, type OpResult, stepNodes } from './graph-core'
import { tidyLayout } from './layout'
import type { NodeData } from './nodes'
import { freshRuntime } from './protocol'
import {
  type Arm,
  defaultConfig,
  type EdgeDef,
  pruneConfig,
  type Scenario,
  type ScenarioStep,
  type StepConfig,
  type StepDef,
  type StepKind
} from './scenario'

/** The graph, reduced to what running it needs. */
export interface RunPlan {
  /** The workflow this run belongs to. A run always has one — it is what the
   *  gateway files the events under and what an ask is routed back to. */
  id: string
  name: string
  scenario?: Scenario
  steps: { id: string; kind: StepKind; config: StepConfig }[]
  edges: { id: string; source: string; target: string; sourceHandle?: string; loop?: boolean }[]
}

/** The executor takes this rather than the graph itself so it never has to know
 *  about React Flow — same reason toScenario exists, and the same boundary. */
export function runPlan(g: Graph, name: string, id: string): RunPlan {
  return {
    id,
    name,
    scenario: toScenario(g),
    steps: stepNodes(g).map(n => {
      const { def, config } = dataOf(n)

      return { id: n.id, kind: def.kind, config }
    }),
    edges: g.edges.map(e => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle ?? undefined,
      loop: isLoop(e)
    }))
  }
}

export function toScenario(g: Graph): Scenario {
  return {
    version: 1,
    steps: stepNodes(g).map((n): ScenarioStep => {
      const { def, config } = dataOf(n)

      return {
        id: n.id,
        kind: def.kind,
        config,
        position: { x: Math.round(n.position.x), y: Math.round(n.position.y) },
        icon: def.icon,
        doing: def.doing
      }
    }),
    // A wire carries no routing of its own any more — it names the arm it
    // leaves by, and the arm travels with the gate's config.
    edges: stepEdges(g).map((e): EdgeDef => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle ?? undefined,
      targetHandle: e.targetHandle ?? undefined,
      loop: isLoop(e)
    }))
  }
}

const stepEdges = (g: Graph) => {
  const ids = new Set(stepNodes(g).map(n => n.id))

  return g.edges.filter(e => ids.has(e.source) && ids.has(e.target))
}

/** Reconcile each gate's outputs with the wires leaving it, before either is
 *  built — a wire names the port it expects and the gate has to actually have
 *  it, or the canvas draws a wire from nowhere.
 *
 *  A payload that declares its outputs is taken at its word, and anything its
 *  wires ask for on top is added. A payload that declares none gets a table
 *  read off the wires, which is what lets an agent send a whole scenario
 *  without stating the routing twice. */
function gateWiring(s: Scenario) {
  const arms = new Map<string, Arm[]>()
  const handles = new Map<string, string>()

  for (const step of s.steps) {
    if (step.kind !== 'gate') {
      continue
    }

    const mine: Arm[] = (step.config.arms ?? []).map(a => ({ ...a }))

    for (const e of s.edges) {
      if (e.source !== step.id) {
        continue
      }

      const id =
        e.sourceHandle ||
        freeArmId(
          mine.map(a => a.id),
          e.loop ? 'loop' : 'pass'
        )

      // Two wires on one id is a fan-out sharing a condition, not a second arm.
      if (!mine.some(a => a.id === id)) {
        mine.push({ id, when: guessWhen(mine, !!e.loop) })
      }

      handles.set(e.id || edgeIdFor(e.source, e.target), id)
    }

    if (mine.length) {
      arms.set(step.id, mine)
    }
  }

  return { arms, handles }
}

/** Rebuild a graph from an authored scenario. Anything the scenario doesn't
 *  place gets laid out, which is what lets an agent send a whole workflow
 *  without knowing the first thing about the canvas's geometry. */
export function fromScenario(s: Scenario): Graph {
  const wires = gateWiring(s)

  const nodes: Node[] = s.steps.map(step => {
    const def: StepDef = {
      id: step.id,
      kind: step.kind,
      title: step.config.title,
      icon: step.icon,
      doing: step.doing,
      profile: step.config.profile,
      model: step.config.model
    }

    // Authored config is an overlay on the kind's defaults, never a
    // replacement, and it's cut to the kind on the way in — a payload from an
    // older schema (or a hand-written one) can otherwise seed every reader
    // with fields the kind stopped having.
    const config: StepConfig = {
      ...defaultConfig(def),
      ...pruneConfig(step.kind, step.config)
    } as StepConfig

    if (step.kind === 'gate') {
      config.arms = wires.arms.get(step.id) ?? config.arms
    }

    return {
      id: step.id,
      type: step.kind,
      position: step.position ?? { x: 0, y: 0 },
      data: { def, config, rt: freshRuntime(), selected: false } satisfies NodeData
    }
  })

  const edges: Edge[] = (s.edges ?? []).map(e => {
    const id = e.id || edgeIdFor(e.source, e.target)
    const sourceHandle = wires.handles.get(id) ?? e.sourceHandle

    return {
      id,
      source: e.source,
      target: e.target,
      ...(sourceHandle ? { sourceHandle } : {}),
      ...(e.targetHandle ? { targetHandle: e.targetHandle } : {}),
      type: 'data',
      data: { state: 'idle', loop: e.loop }
    }
  })

  const placed = s.steps.every(step => step.position) ? nodes : tidyLayout(nodes, edges)

  return { nodes: placed, edges }
}

/** Swap the whole scenario for another one. The counterpart to graph_get: an
 *  agent that wants to author a workflow outright shouldn't have to express it
 *  as thirty surgical edits. */
export function setScenario(g: Graph, s: Scenario): OpResult {
  return {
    ok: true,
    graph: fromScenario({ ...s, version: 1, steps: s?.steps ?? [], edges: s?.edges ?? [] }),
    message: `Replaced the scenario — ${s?.steps?.length ?? 0} steps, ${s?.edges?.length ?? 0} wires.`,
    edit: `scenario · ${s?.steps?.length ?? 0} steps`
  }
}
