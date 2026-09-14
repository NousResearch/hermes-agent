import { type BitReader, type BitWriter, createLoadout, Dict, idxOf, indexBits, LoadoutError } from '@/lib/loadout'
import type { StarmapEdge, StarmapGraph, StarmapNode } from '@/types/hermes'

import { isProviderSource } from './sources'

// ── Star-map share code ───────────────────────────────────────────────────────
//
// The body schema for a star map, riding the generic loadout codec (@/lib/loadout
// owns the bitstream, DEFLATE, version+checksum frame, and base64url). We encode
// what the map RENDERS — each node's kind, its time POSITION (12-bit quantized,
// not an absolute epoch), radius inputs (useCount/state/pinned), and an interned
// label + category — plus edges as fixed-width node indices. Memory prose is
// dropped; labels are trimmed. DEFLATE then makes the repetitive label/category
// text almost free. A 60-skill map is a few hundred chars.

const VERSION = 4 // v4: provider memory sources (dict-backed 'other' slot)
const PREFIX = 'HML' // "Hermes Memory Loadout" — namespaces our codes like WoW's leading bytes.
const MAX_LABEL = 64 // trim runaway memory titles so one card can't bloat the code.

const trim = (s: string): string => (s.length > MAX_LABEL ? s.slice(0, MAX_LABEL) : s)

const KINDS = ['skill', 'memory'] as const
const STATES = ['active', 'archived', 'disabled', 'draft'] as const
// 2-bit slot. 'other' = the source is a memory-provider name ('honcho', …),
// carried as a dict back-reference right after the slot (v4+). v3 encoders
// never wrote index 3, so the slot stays decodable across versions.
const MEM_SOURCES = ['none', 'memory', 'profile', 'other'] as const
const OTHER_SOURCE = 3
const CREATED_BY = ['none', 'agent', 'user'] as const

const REC_BITS = 12 // time position resolution: 1/4096 of the span — sub-pixel here.
const REC_MAX = (1 << REC_BITS) - 1

const finiteTs = (v?: null | number): null | number =>
  typeof v === 'number' && Number.isFinite(v) ? Math.max(0, Math.round(v)) : null

function writeNode(w: BitWriter, n: StarmapNode, dict: Dict, minTs: number, span: number): void {
  w.uint(idxOf(KINDS, n.kind), 1)
  w.varint(dict.id(trim(n.label || '')))
  w.varint(dict.id(n.category || ''))
  w.varint(Math.max(0, n.useCount | 0))
  w.uint(idxOf(STATES, n.state), 2)

  // Fixed sources use the enum slot; a provider name rides the string dict
  // behind the 'other' sentinel (a provider literally named 'other' included).
  const src = n.memorySource ?? 'none'

  if (isProviderSource(src)) {
    w.uint(OTHER_SOURCE, 2)
    w.varint(dict.id(src))
  } else {
    w.uint(idxOf(MEM_SOURCES, src), 2)
  }

  w.uint(idxOf(CREATED_BY, n.createdBy ?? 'none'), 2)
  w.bit(n.pinned)

  // Time as a 12-bit POSITION within [minTs, maxTs] — not an absolute epoch.
  const ts = finiteTs(n.timestamp)

  if (ts === null) {
    w.bit(0)
  } else {
    w.bit(1)
    w.uint(span > 0 ? Math.round(((ts - minTs) / span) * REC_MAX) : 0, REC_BITS)
  }
}

function readNode(r: BitReader, dict: string[], i: number, minTs: number, span: number, version: number): StarmapNode {
  const kind = KINDS[r.uint(1)] ?? 'skill'
  const label = dict[r.varint()] ?? ''
  const category = dict[r.varint()] ?? ''
  const useCount = r.varint()
  const state = STATES[r.uint(2)] ?? 'active'
  const memSlot = r.uint(2)
  // v4+: slot 3 means "provider name follows as a dict back-reference".
  // v3 never wrote slot 3, and carries no varint after the slot.
  const memSrc = version >= 4 && memSlot === OTHER_SOURCE ? (dict[r.varint()] ?? 'memory') : (MEM_SOURCES[memSlot] ?? 'none')
  const createdBy = CREATED_BY[r.uint(2)] ?? 'none'
  const pinned = r.bit() === 1
  const timestamp = r.bit() === 1 ? minTs + (span > 0 ? Math.round((r.uint(REC_BITS) / REC_MAX) * span) : 0) : null

  // Ids are synthesized (they're never displayed); memory ids mirror the scan's
  // `memory:<source>:<index>` shape so the rest of the UI is none the wiser.
  const isMemory = kind === 'memory'
  const source = memSrc === 'none' ? 'memory' : memSrc

  return {
    category,
    createdBy: createdBy === 'none' ? null : createdBy,
    id: isMemory ? `memory:${source}:${i}` : `s${i}`,
    kind,
    label,
    memorySource: isMemory ? source : undefined,
    pinned,
    state,
    timestamp,
    useCount
  }
}

function writeGraph(w: BitWriter, graph: StarmapGraph): void {
  const dict = new Dict()

  // Intern labels + categories (and any provider source names — writeNode
  // emits them as dict back-references) BEFORE the dict is serialized.
  for (const n of graph.nodes) {
    dict.id(trim(n.label || ''))
    dict.id(n.category || '')

    if (isProviderSource(n.memorySource)) {
      dict.id(n.memorySource)
    }
  }

  const stamps = graph.nodes.map(n => finiteTs(n.timestamp)).filter((v): v is number => v !== null)
  const minTs = stamps.length ? Math.min(...stamps) : 0
  const maxTs = stamps.length ? Math.max(...stamps) : 0
  const span = maxTs - minTs

  w.varint(minTs)
  w.varint(maxTs)
  w.varint(dict.list.length)

  for (const s of dict.list) {
    w.str(s)
  }

  w.varint(graph.nodes.length)

  for (const n of graph.nodes) {
    writeNode(w, n, dict, minTs, span)
  }

  // Edges reference nodes by position; drop any whose endpoints aren't both nodes.
  const order = new Map(graph.nodes.map((n, i) => [n.id, i]))
  const edges = graph.edges.filter(e => order.has(e.source) && order.has(e.target))
  const bits = indexBits(graph.nodes.length)
  w.varint(edges.length)

  for (const e of edges) {
    w.uint(order.get(e.source)!, bits)
    w.uint(order.get(e.target)!, bits)
  }
}

// The body reader, parameterized by schema version: v3 has no dict-backed
// provider sources; v4 adds them. Registered once as `read` (current) and
// once per legacy version below, so old shared codes keep loading.
const readGraph =
  (version: number) =>
  (r: BitReader): StarmapGraph => {
    const minTs = r.varint()
    const maxTs = r.varint()
    const span = maxTs - minTs

    const dictLen = r.varint()
    const dict: string[] = []

    for (let i = 0; i < dictLen; i += 1) {
      dict.push(r.str())
    }

    const nodeCount = r.varint()
    const nodes: StarmapNode[] = []

    for (let i = 0; i < nodeCount; i += 1) {
      nodes.push(readNode(r, dict, i, minTs, span, version))
    }

    const bits = indexBits(nodeCount)
    const edgeCount = r.varint()
    const edges: StarmapEdge[] = []

    for (let i = 0; i < edgeCount; i += 1) {
      const src = nodes[r.uint(bits)]
      const dst = nodes[r.uint(bits)]

      if (src && dst) {
        edges.push({ source: src.id, target: dst.id })
      }
    }

    const counts = new Map<string, number>()

    for (const n of nodes) {
      counts.set(n.category, (counts.get(n.category) ?? 0) + 1)
    }

    const clusters = [...counts.entries()]
      .map(([category, count]) => ({ category, count }))
      .sort((a, b) => b.count - a.count)

    // Memory cards are dropped (viz-only); a marker lets the UI tell a decoded map
    // apart from a freshly-scanned one.
    return { clusters, edges, memory: [], nodes, stats: { imported: true } }
  }

export class ShareCodeError extends LoadoutError {}

const codec = createLoadout<StarmapGraph>({
  error: ShareCodeError,
  legacyReads: { 3: readGraph(3) },
  noun: 'map code',
  prefix: PREFIX,
  read: readGraph(VERSION),
  version: VERSION,
  write: writeGraph
})

// Serialize a star-map graph to a short, opaque, clipboard-safe loadout string.
export function encodeShareCode(graph: StarmapGraph): string {
  return codec.encode(graph)
}

// Parse a loadout string back into a (viz-complete, text-synthesized) graph.
export function decodeShareCode(code: string): StarmapGraph {
  return codec.decode(code)
}
