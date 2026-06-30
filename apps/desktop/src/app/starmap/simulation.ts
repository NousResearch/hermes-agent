import { forceCollide, forceLink, forceManyBody, forceRadial, forceSimulation, type Simulation } from 'd3-force'

import type { StarmapGraph, StarmapNode } from '@/types/hermes'

import { RING_STEPS } from './constants'
import { clamp, hash, nodeRadius, radiusForRecency } from './geometry'
import { formatDate } from './text'
import { computeRecency, recForRatio } from './time-axis'
import type { Ring, SimLink, SimNode } from './types'

export interface BuiltSim {
  byId: Map<string, SimNode>
  links: SimLink[]
  nodes: SimNode[]
  rings: Ring[]
  sim: Simulation<SimNode, SimLink>
}

const DAY = 86_400

// Constant ring SCALE: the core radius and the per-ring band are pinned to the
// canonical 5-ring layout, so the empty core and every band are ALWAYS that
// size on the disk — more data grows the disk OUTWARD (more rings) instead of
// stretching a fixed disk thinner. The camera caps its zoom at the 5-ring
// extent (see fitViewport), so this world size is also a constant screen size.
const RING_CORE = radiusForRecency(recForRatio(0))
const RING_BAND = (radiusForRecency(recForRatio(1)) - RING_CORE) / RING_STEPS
const ringRadius = (i: number): number => RING_CORE + i * RING_BAND

// Place a node INSIDE its ring's band (the annulus the ring caps), biased toward
// mid-band so it reads as "within the ring", only occasionally grazing an edge —
// never sitting on the outline like a bead.
const placeRadius = (i: number, id: string): number => {
  const outer = ringRadius(i)
  const inner = i > 0 ? ringRadius(i - 1) : RING_CORE - RING_BAND * 0.5
  const h = (hash(id) % 1000) / 1000

  return outer - (0.15 + 0.7 * h) * (outer - inner)
}

interface Unit {
  kind: 'day' | 'month'
  step: number
}

// "Nice" calendar intervals, fine → coarse, WITH intermediate rungs (2-day,
// bi-weekly, 2/3/6-month) so the bucketer can land NEAR the target count instead
// of jumping straight from weekly (≈9) to monthly (≈3) and missing the ~5 sweet spot.
const UNITS: Unit[] = [
  { kind: 'day', step: 1 },
  { kind: 'day', step: 2 },
  { kind: 'day', step: 7 },
  { kind: 'day', step: 14 },
  { kind: 'month', step: 1 },
  { kind: 'month', step: 2 },
  { kind: 'month', step: 3 },
  { kind: 'month', step: 6 },
  { kind: 'month', step: 12 }
]

// Floor a timestamp to the start of its calendar bucket — the key nodes group by.
function bucketStart(ts: number, { kind, step }: Unit): number {
  if (kind === 'day') {
    const period = step * DAY

    return Math.floor(ts / period) * period
  }

  const d = new Date(ts * 1000)
  d.setUTCHours(0, 0, 0, 0)
  // Floor to a step-month boundary in ABSOLUTE months so steps align across
  // years (3-month → Jan/Apr/Jul/Oct, 12-month → Jan).
  const absMonth = Math.floor((d.getUTCFullYear() * 12 + d.getUTCMonth()) / step) * step
  d.setUTCFullYear(Math.floor(absMonth / 12), absMonth % 12, 1)

  return Math.floor(d.getTime() / 1000)
}

const populatedStarts = (stamps: number[], u: Unit): number[] => [...new Set(stamps.map(t => bucketStart(t, u)))].sort((a, b) => a - b)

// "Nice ticks" for time (à la D3/Heckbert): aim for a target ring count that
// grows ~log2 with the span, then snap to the calendar interval whose POPULATED
// count lands nearest it (ties + overshoot break toward fewer/finer). The floor
// is 5 — fewer than that and the play-through "steps" between rings get big and
// abrupt; ~5+ evenly-paced rings give the smooth Spore-style build-up.
function chooseUnit(stamps: number[], spanDays: number): Unit {
  const target = clamp(Math.round(4 + Math.log2(Math.max(1, spanDays / 60))), 5, 12)
  let best = UNITS[0]!
  let bestScore = Infinity

  for (const u of UNITS) {
    const count = populatedStarts(stamps, u).length

    if (!count) {
      continue
    }

    const score = Math.abs(count - target) + (count > target ? 0.5 : 0)

    if (score < bestScore) {
      bestScore = score
      best = u
    }
  }

  return best
}

function bucketLabel(ts: number, { kind, step }: Unit): string {
  if (kind === 'day') {
    return formatDate(ts)
  }

  try {
    const d = new Date(ts * 1000)

    return step >= 12 ? String(d.getUTCFullYear()) : d.toLocaleDateString(undefined, { month: 'short', timeZone: 'UTC', year: 'numeric' })
  } catch {
    return formatDate(ts)
  }
}

interface Layout {
  // bucket/cap ring a node belongs to (the ring it ignites behind)
  index: (n: StarmapNode) => number
  // reveal coordinate (0–1) a node ignites at — staggered within its band
  rec: (n: StarmapNode) => number
  rings: Ring[]
  // world radius a node is drawn at (inside its band)
  tr: (n: StarmapNode) => number
}

// Even, unlabeled-ish fallback when there's no usable time span (undated graph
// or one instant): keep the legacy continuous mapping so nothing regresses.
function evenLayout(recById: Map<string, number>, minTs: null | number, maxTs: null | number, timed: boolean): Layout {
  const rings: Ring[] = Array.from({ length: RING_STEPS + 1 }, (_, i) => ({
    label: timed && minTs !== null && maxTs !== null ? formatDate(Math.round(minTs + (maxTs - minTs) * (i / RING_STEPS))) : null,
    r: ringRadius(i),
    ratio: recForRatio(i / RING_STEPS)
  }))

  const capRing = (rec: number): number => {
    for (let i = 0; i < rings.length; i += 1) {
      if ((rings[i]?.ratio ?? 1) >= rec - 1e-3) {
        return i
      }
    }

    return rings.length - 1
  }

  return {
    index: n => capRing(recById.get(n.id) ?? 0),
    rec: n => recById.get(n.id) ?? 0,
    rings,
    tr: n => radiusForRecency(recById.get(n.id) ?? 0)
  }
}

// One equal-width ring per POPULATED calendar bucket; a bucket's nodes fill the
// band INSIDE their ring (fanned by angle) and ignite staggered across it.
function buildLayout(graph: StarmapGraph, recById: Map<string, number>, minTs: null | number, maxTs: null | number, timed: boolean): Layout {
  const stamps = graph.nodes.map(n => Number(n.timestamp)).filter(Number.isFinite)

  if (!(timed && minTs !== null && maxTs !== null && maxTs > minTs && stamps.length)) {
    return evenLayout(recById, minTs, maxTs, timed)
  }

  const span = maxTs - minTs
  const unit = chooseUnit(stamps, span / DAY)
  const starts = populatedStarts(stamps, unit)

  if (starts.length < 2) {
    return evenLayout(recById, minTs, maxTs, timed)
  }

  const indexOfStart = new Map(starts.map((s, i) => [s, i]))
  // Reveal pacing is per-BUCKET (uniform), matching the equal-width bands: each
  // ring is one even step. (Radius is already index-based.) Using raw time here
  // decouples a ring's ignite moment from its position — a bursty gap makes a
  // ring appear bands ahead of the nodes that belong to it. Labels stay real dates.
  const last = Math.max(1, starts.length - 1)
  const rings: Ring[] = starts.map((s, i) => ({ label: bucketLabel(s, unit), r: ringRadius(i), ratio: recForRatio(i / last) }))

  // A node's bucket is its ring; undated nodes (rare, in an otherwise-timed
  // graph) fall to the newest ring so they still appear.
  const indexFor = (n: StarmapNode): number => {
    const ts = Number(n.timestamp)

    return Number.isFinite(ts) ? (indexOfStart.get(bucketStart(ts, unit)) ?? starts.length - 1) : starts.length - 1
  }

  // Node POSITION fills the band inside its bucket ring (placeRadius); its IGNITE
  // time is staggered ACROSS that band, ordered by real timestamp, so a busy
  // bucket trickles in over its whole slice instead of every node popping at
  // once (the "everything floods in at the end" bug).
  const buckets: StarmapNode[][] = starts.map(() => [])

  for (const n of graph.nodes) {
    buckets[indexFor(n)]!.push(n)
  }

  const tsOf = (n: StarmapNode): number => (Number.isFinite(Number(n.timestamp)) ? Number(n.timestamp) : Infinity)
  const recByNode = new Map<string, number>()

  buckets.forEach((bucket, i) => {
    bucket.sort((a, b) => (tsOf(a) === tsOf(b) ? a.id.localeCompare(b.id) : tsOf(a) - tsOf(b)))

    const hi = rings[i]!.ratio
    const lo = i > 0 ? rings[i - 1]!.ratio : 0
    const m = bucket.length

    // f ∈ (0,1]: first node lands just inside the band, last node ON the ring.
    bucket.forEach((n, k) => recByNode.set(n.id, lo + ((k + 1) / m) * (hi - lo)))
  })

  return {
    index: indexFor,
    rec: n => recByNode.get(n.id) ?? 0,
    rings,
    tr: n => placeRadius(indexFor(n), n.id)
  }
}

// Build the radial time simulation: a node's distance from the core encodes its
// timestamp bucket (radial force dominates; charge/collide only spread nodes
// around their date ring). Rings are dated, equal-width gridlines.
export function buildSimulation(graph: StarmapGraph, onTick: () => void): BuiltSim {
  const { maxTs, minTs, rec: recById, timed } = computeRecency(graph.nodes)
  const { index, rec: recOf, rings, tr: trOf } = buildLayout(graph, recById, minTs, maxTs, timed)

  const nodes: SimNode[] = graph.nodes.map(n => {
    const rec = recOf(n)
    const tr = trOf(n)
    const angle = ((hash(n.id) % 3600) / 3600) * Math.PI * 2

    return { ...n, outerRingIndex: index(n), rec, tr, vx: 0, vy: 0, x: Math.cos(angle) * tr, y: Math.sin(angle) * tr }
  })

  const byId = new Map(nodes.map(n => [n.id, n]))

  const links: SimLink[] = graph.edges
    .filter(e => byId.has(e.source) && byId.has(e.target))
    .map(e => ({ source: e.source, target: e.target }))

  const sim = forceSimulation(nodes)
    .alphaDecay(0.05)
    .velocityDecay(0.62)
    .force('charge', forceManyBody<SimNode>().strength(-12))
    .force(
      'link',
      forceLink<SimNode, SimLink>(links)
        .id(n => n.id)
        .distance(26)
        .strength(0.06)
    )
    .force(
      'collide',
      forceCollide<SimNode>()
        .radius(n => nodeRadius(n) + 2)
        .iterations(2)
    )
    .force('radial', forceRadial<SimNode>(n => (n as SimNode).tr, 0, 0).strength(0.92))
    .on('tick', onTick)

  return { byId, links, nodes, rings, sim }
}
