import type { StarmapGraph, StarmapNode } from '@/types/hermes'

import { clamp } from './geometry'

// Empty lead-in: push the oldest node off 0 so the timeline opens on a beat of
// emptiness (you watch the first node grow in). Radial position is otherwise a
// truthful linear map of time, so rings line up with the nodes they date.
export const LEAD_IN = 0.06
export const recForRatio = (ratio: number): number => LEAD_IN + (1 - LEAD_IN) * clamp(ratio, 0, 1)

export interface Recency {
  maxTs: null | number
  minTs: null | number
  // id → recency ratio (0 oldest … 1 newest). Timed by timestamp when the span
  // is real, else ordinal so an undated graph still "builds up" in a stable order.
  rec: Map<string, number>
  timed: boolean
}

// Shared recency model for both the radial layout (simulation.ts) and the
// timeline scrubber, so a node's ring distance and its ignite time agree.
export function computeRecency(nodes: StarmapNode[]): Recency {
  const known = nodes
    .map(n => (typeof n.timestamp === 'number' && Number.isFinite(n.timestamp) ? Number(n.timestamp) : null))
    .filter((v): v is number => v !== null)

  const minTs = known.length ? Math.min(...known) : null
  const maxTs = known.length ? Math.max(...known) : null
  const timed = minTs !== null && maxTs !== null && maxTs > minTs

  const ordered = [...nodes].sort((a, b) => {
    const at = typeof a.timestamp === 'number' ? a.timestamp : Infinity
    const bt = typeof b.timestamp === 'number' ? b.timestamp : Infinity

    return at === bt ? a.id.localeCompare(b.id) : at - bt
  })

  const ordRatio = new Map(ordered.map((n, i) => [n.id, ordered.length > 1 ? i / (ordered.length - 1) : 0]))
  const rec = new Map<string, number>()

  // Radius is a truthful linear map of time (ordinal only as a fallback for the
  // undated). Co-timed nodes share a radius and fan out by ANGLE in the sim — so
  // a burst reads as a populated ring, and the dated rings stay accurate.
  for (const n of nodes) {
    const ratio =
      timed && typeof n.timestamp === 'number' && minTs !== null && maxTs !== null
        ? (Number(n.timestamp) - minTs) / (maxTs - minTs)
        : (ordRatio.get(n.id) ?? 0)

    rec.set(n.id, recForRatio(ratio))
  }

  return { maxTs, minTs, rec, timed }
}

export interface TimeBucket {
  memory: number
  skill: number
  total: number
}

export interface TimeAxis {
  buckets: TimeBucket[]
  maxTotal: number
  maxTs: null | number
  minTs: null | number
  // Total node count — the denominator for the "n / total" label when undated.
  size: number
  timed: boolean
}

// Bucket nodes across recency [0,1] into a fixed-width histogram — the little
// bars the scrubber rides over. Skill/memory kept separate so the bars can show
// the same two-tone split as the map glyphs.
export function buildTimeAxis(graph: StarmapGraph, bucketCount = 48): TimeAxis {
  const { maxTs, minTs, rec, timed } = computeRecency(graph.nodes)
  const n = Math.max(1, bucketCount)
  const buckets: TimeBucket[] = Array.from({ length: n }, () => ({ memory: 0, skill: 0, total: 0 }))

  for (const node of graph.nodes) {
    const r = rec.get(node.id) ?? 0
    const idx = clamp(Math.floor(r * n), 0, n - 1)
    const b = buckets[idx]!
    b.total += 1

    if (node.kind === 'memory') {
      b.memory += 1
    } else {
      b.skill += 1
    }
  }

  const maxTotal = buckets.reduce((m, b) => Math.max(m, b.total), 0)

  return { buckets, maxTotal, maxTs, minTs, size: graph.nodes.length, timed }
}

// Wall-clock date at a reveal ratio (linear in time when the graph is dated).
export function dateAtReveal(axis: TimeAxis, reveal: number): null | number {
  if (!axis.timed || axis.minTs === null || axis.maxTs === null) {
    return null
  }

  return Math.round(axis.minTs + clamp(reveal, 0, 1) * (axis.maxTs - axis.minTs))
}
