// Arms — a gate's outputs.
//
// The gate owns them; a wire only names one. Everything that used to read a
// condition off an edge reads it off the arm the edge leaves by, so an arm can
// sit there unwired while you decide where it goes.

import type { Edge, Node } from '@xyflow/react'

import { dataOf, fail, type Graph, type OpResult, resolveStep } from './graph-core'
import type { NodeData } from './nodes'
import { type Arm, describePredicate, type Predicate } from './scenario'

export const armsOf = (g: Graph, gateId: string): Arm[] =>
  (g.nodes.find(n => n.id === gateId)?.data as NodeData | undefined)?.config.arms ?? []

/** What the canvas prints beside an output: the name you gave it, or the
 *  condition itself when you haven't. */
export const armLabel = (a: Arm) => a.label?.trim() || describePredicate(a.when)

/** Rewrite a gate's arms in place. */
export function withArms(g: Graph, gateId: string, arms: Arm[]): Node[] {
  return g.nodes.map(n =>
    n.id === gateId ? { ...n, data: { ...n.data, config: { ...(n.data as NodeData).config, arms } } } : n
  )
}

/** The first unclaimed id in the `pass`, `pass_2`, `pass_3` series. Appends
 *  what it hands out, so a caller naming several arms in one pass doesn't have
 *  to rebuild the list between them. */
export function freeArmId(taken: string[], wanted: string): string {
  let name = wanted

  for (let i = 2; taken.includes(name); i++) {
    name = `${wanted}_${i}`
  }

  taken.push(name)

  return name
}

/** The condition a brand-new arm starts on. A gate's first forward output is
 *  the happy path, so "all pass" is the useful guess; anything after it is the
 *  else, because first match wins and a second "all pass" would sit there
 *  unreachable behind the first. A rework arm tests the opposite. */
export function guessWhen(existing: Arm[], loop: boolean): Predicate {
  if (loop) {
    return { mode: 'any-fail' }
  }

  return existing.some(a => a.when.mode !== 'any-fail') ? { mode: 'always' } : { mode: 'all-pass' }
}

/** Add an output to a gate. Unwired — which is the whole point: you can lay
 *  out a routing table and then decide where each arm goes. */
export function addArm(g: Graph, ref: string, when?: Predicate, label?: string): OpResult {
  const gate = resolveStep(g, ref)

  if (!gate) {
    return fail(g, `There's no step called "${ref}".`)
  }

  if (dataOf(gate).def.kind !== 'gate') {
    return fail(g, `${gate.id} isn't a gate, so it has one output, not a table of them.`)
  }

  const arms = armsOf(g, gate.id)

  const arm: Arm = {
    id: freeArmId(
      arms.map(a => a.id),
      'pass'
    ),
    when: when ?? guessWhen(arms, false),
    ...(label ? { label } : {})
  }

  return {
    ok: true,
    graph: { nodes: withArms(g, gate.id, [...arms, arm]), edges: g.edges },
    message: `Added an output to ${gate.id}: "${armLabel(arm)}". Wire it to say where it goes.`,
    edit: `${gate.id} · + ${armLabel(arm)}`
  }
}

/** Drop an output, and whatever was wired to it — the wire has no port to
 *  leave by once the arm is gone. */
export function removeArm(g: Graph, ref: string, armId: string): OpResult {
  const gate = resolveStep(g, ref)

  if (!gate) {
    return fail(g, `There's no step called "${ref}".`)
  }

  const arms = armsOf(g, gate.id)
  const arm = arms.find(a => a.id === armId)

  if (!arm) {
    return fail(g, `${gate.id} has no "${armId}" output.`)
  }

  return {
    ok: true,
    graph: {
      nodes: withArms(
        g,
        gate.id,
        arms.filter(a => a.id !== armId)
      ),
      edges: g.edges.filter(e => !(e.source === gate.id && e.sourceHandle === armId))
    },
    message: `Dropped ${gate.id}'s "${armLabel(arm)}" output.`,
    edit: `${gate.id} · − ${armLabel(arm)}`
  }
}

/** Edit one output on a gate: its condition, its name, or both. */
export function setBranch(g: Graph, ref: string, armId: string, patch: { when?: Predicate; label?: string }): OpResult {
  const gate = resolveStep(g, ref)

  if (!gate) {
    return fail(g, `There's no step called "${ref}".`)
  }

  const arms = armsOf(g, gate.id)

  if (!arms.some(a => a.id === armId)) {
    return fail(g, `${gate.id} has no "${armId}" output.`)
  }

  const next = arms.map(a => (a.id === armId ? { ...a, ...patch } : a))
  const arm = next.find(a => a.id === armId)!

  return {
    ok: true,
    graph: { nodes: withArms(g, gate.id, next), edges: g.edges },
    message: `Updated ${gate.id}'s "${armLabel(arm)}" output.`,
    edit: `${gate.id} · ${armLabel(arm)}`
  }
}

/** Where an arm's wires go, if it has any. An arm with none is a rule you've
 *  written down and not yet pointed anywhere — legal, and flagged by check. */
export const armTargets = (g: Graph, gateId: string, armId: string): Edge[] =>
  g.edges.filter(e => e.source === gateId && (e.sourceHandle ?? '') === armId)
