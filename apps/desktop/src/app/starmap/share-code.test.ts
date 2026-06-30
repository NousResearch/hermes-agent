import { describe, expect, it } from 'vitest'

import type { StarmapGraph } from '@/types/hermes'

import { decodeShareCode, encodeShareCode, ShareCodeError } from './share-code'

function sampleGraph(): StarmapGraph {
  return {
    clusters: [],
    edges: [
      { source: 'skill-a', target: 'skill-b' },
      { source: 'skill-b', target: 'memory:profile:0' }
    ],
    memory: [
      { body: 'Prefers concise answers.', source: 'profile', timestamp: 1_700_000_000, title: 'Tone' },
      { body: 'Uses a worktree.', source: 'memory', timestamp: null, title: 'Env' }
    ],
    nodes: [
      { category: 'devops', createdBy: 'agent', id: 'skill-a', kind: 'skill', label: 'skill-a', pinned: true, state: 'active', timestamp: 1_699_900_000, useCount: 7 },
      { category: 'devops', createdBy: null, id: 'skill-b', kind: 'skill', label: 'skill-b', pinned: false, state: 'draft', timestamp: 1_699_950_000, useCount: 0 },
      { category: 'memory', createdBy: null, id: 'memory:profile:0', kind: 'memory', label: 'A fact', memorySource: 'profile', pinned: false, state: 'active', timestamp: 1_700_000_000, useCount: 0 }
    ],
    stats: {}
  }
}

// Decoded edges compared by node POSITION (ids are synthesized), so topology is
// the invariant, not the literal id strings.
const topology = (g: StarmapGraph): [number, number][] => {
  const idx = new Map(g.nodes.map((n, i) => [n.id, i]))

  return g.edges.map(e => [idx.get(e.source)!, idx.get(e.target)!])
}

describe('share-code', () => {
  // The viz contract: everything the star map RENDERS survives — kinds, radius
  // inputs, time position, edge topology — while text is dropped (it's a loadout,
  // not a backup).
  it('preserves the visualization', () => {
    const g = sampleGraph()
    const decoded = decodeShareCode(encodeShareCode(g))
    const span = 1_700_000_000 - 1_699_900_000
    const tol = Math.ceil(span / 4095) + 1

    expect(decoded.nodes).toHaveLength(g.nodes.length)

    decoded.nodes.forEach((d, i) => {
      const o = g.nodes[i]!
      expect(d.kind).toBe(o.kind)
      expect(d.label).toBe(o.label)
      expect(d.useCount).toBe(o.useCount)
      expect(d.state).toBe(o.state)
      expect(d.pinned).toBe(o.pinned)
      expect(d.category).toBe(o.category)
      expect(d.memorySource).toBe(o.memorySource)
      expect(d.createdBy).toBe(o.createdBy)

      if (o.timestamp == null) {
        expect(d.timestamp).toBeNull()
      } else {
        expect(Math.abs((d.timestamp ?? 0) - o.timestamp)).toBeLessThanOrEqual(tol)
      }
    })

    expect(topology(decoded)).toEqual(topology(g))
  })

  it('drops memory prose (loadout is viz-only)', () => {
    expect(decodeShareCode(encodeShareCode(sampleGraph())).memory).toHaveLength(0)
  })

  it('rebuilds clusters from node categories', () => {
    const decoded = decodeShareCode(encodeShareCode(sampleGraph()))

    expect(decoded.clusters.find(c => c.category === 'devops')?.count).toBe(2)
  })

  it('produces a short, opaque, prefixed code', () => {
    const code = encodeShareCode(sampleGraph())

    expect(code.startsWith('HML')).toBe(true)
    expect(code.slice(3)).toMatch(/^[A-Za-z0-9_-]+$/)
    // Strictly smaller than the naive JSON it replaces — the whole point.
    expect(code.length).toBeLessThan(JSON.stringify(sampleGraph()).length)
  })

  it('stays compact on a large graph (no string bloat)', () => {
    const nodes = Array.from({ length: 500 }, (_, i) => ({
      category: `cat-${i % 8}`,
      createdBy: 'agent' as const,
      id: `s${i}`,
      kind: 'skill' as const,
      label: `A fairly verbose skill label number ${i}`,
      pinned: false,
      state: 'active',
      timestamp: 1_700_000_000 + i * 3600,
      useCount: i % 50
    }))

    const graph: StarmapGraph = { clusters: [], edges: [], memory: [], nodes, stats: {} }
    const code = encodeShareCode(graph)

    // Deflate keeps even verbose, repetitive labels far below the naive JSON.
    expect(code.length).toBeLessThan(JSON.stringify(graph).length / 5)
  })

  it('handles an empty graph', () => {
    const decoded = decodeShareCode(encodeShareCode({ clusters: [], edges: [], memory: [], nodes: [], stats: {} }))

    expect(decoded.nodes).toHaveLength(0)
    expect(decoded.edges).toHaveLength(0)
  })

  it('drops edges whose endpoints are missing', () => {
    const g = sampleGraph()
    g.edges.push({ source: 'skill-a', target: 'does-not-exist' })

    expect(decodeShareCode(encodeShareCode(g)).edges).toHaveLength(2)
  })

  it('rejects garbage with a ShareCodeError', () => {
    expect(() => decodeShareCode('not a real code !!!')).toThrow(ShareCodeError)
    expect(() => decodeShareCode('')).toThrow(ShareCodeError)
  })

  it('rejects a corrupted (bit-flipped) code', () => {
    const code = encodeShareCode(sampleGraph())
    // Flip a mid-payload char (trailing base64 bits can be dropped on decode).
    const i = Math.floor(code.length / 2)
    const corrupted = code.slice(0, i) + (code[i] === 'A' ? 'B' : 'A') + code.slice(i + 1)

    expect(() => decodeShareCode(corrupted)).toThrow(ShareCodeError)
  })

  it('tolerates whitespace, including internal wraps', () => {
    const code = encodeShareCode(sampleGraph())
    const wrapped = `  ${code.slice(0, 20)}\n${code.slice(20)}\t`

    expect(() => decodeShareCode(wrapped)).not.toThrow()
    expect(decodeShareCode(wrapped).nodes).toHaveLength(sampleGraph().nodes.length)
  })
})
