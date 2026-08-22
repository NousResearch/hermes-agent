import { describe, expect, it } from 'vitest'

import type { StarmapGraph, StarmapNode } from '@/types/hermes'

import {
  conclusionsEnabled,
  distinctOrigins,
  effectiveRange,
  EMPTY_FILTERS,
  filterNodes,
  hasActiveNarrowing,
  isConclusion,
  nodeOrigin
} from './search'

const node = (over: Partial<StarmapNode> & { id: string }): StarmapNode => ({
  category: 'memory',
  createdBy: 'memory',
  kind: 'memory',
  label: over.id,
  pinned: false,
  state: 'active',
  timestamp: 1_700_000_000,
  useCount: 0,
  ...over
})

const graph = (
  nodes: StarmapNode[],
  memory: StarmapGraph['memory'] = [],
  memoryProvider: null | string = null
): StarmapGraph => ({
  clusters: [],
  edges: [],
  memory,
  memoryProvider,
  nodes,
  stats: {}
})

describe('nodeOrigin / distinctOrigins', () => {
  it('defaults to hermes and normalizes case', () => {
    expect(nodeOrigin(node({ id: 'a' }))).toBe('hermes')
    expect(nodeOrigin(node({ id: 'b', origin: 'ChatGPT' }))).toBe('chatgpt')
  })

  it('lists hermes first, imports alphabetically — open-ended for future sources', () => {
    const origins = distinctOrigins([
      node({ id: 'a', origin: 'gemini' }),
      node({ id: 'b' }),
      node({ id: 'c', origin: 'chatgpt' }),
      node({ id: 'd', origin: 'chatgpt' })
    ])

    expect(origins).toEqual(['hermes', 'chatgpt', 'gemini'])
  })
})

describe('filterNodes', () => {
  const g = graph(
    [
      node({ id: 'memory:honcho:0', label: 'garden plan…', origin: 'chatgpt', timestamp: 1_600_000_000 }),
      node({ id: 'memory:memory:1', label: 'DGX cluster facts', timestamp: 1_700_000_000 }),
      node({
        category: 'devops',
        createdBy: 'agent',
        id: 'deploy-skill',
        kind: 'skill',
        label: 'deploy-skill',
        timestamp: 1_650_000_000
      }),
      node({ id: 'memory:honcho:2', label: 'undated note', timestamp: null })
    ],
    [
      { body: 'full tomato planting schedule for clay soil', source: 'honcho', title: 'garden plan…' },
      { body: 'spark cluster', source: 'memory', title: 'DGX cluster facts' }
    ] as StarmapGraph['memory']
  )

  it('returns everything chronologically when nothing narrows (undated last)', () => {
    const ids = filterNodes(g, '', EMPTY_FILTERS).map(n => n.id)

    expect(ids).toEqual(['memory:honcho:0', 'deploy-skill', 'memory:memory:1', 'memory:honcho:2'])
  })

  it('matches memory card BODIES, not just truncated labels', () => {
    const ids = filterNodes(g, 'tomato clay', EMPTY_FILTERS).map(n => n.id)

    expect(ids).toEqual(['memory:honcho:0'])
  })

  it('filters by kind, source, and date range', () => {
    expect(filterNodes(g, '', { ...EMPTY_FILTERS, kind: 'skill' }).map(n => n.id)).toEqual(['deploy-skill'])
    expect(filterNodes(g, '', { ...EMPTY_FILTERS, source: 'chatgpt' }).map(n => n.id)).toEqual(['memory:honcho:0'])
    // 2021-06-06 ≈ 1622930400 — excludes the 2020 chatgpt node, keeps 2022+2023; undated drops.
    expect(filterNodes(g, '', { ...EMPTY_FILTERS, from: '2021-06-06' }).map(n => n.id)).toEqual([
      'deploy-skill',
      'memory:memory:1'
    ])
  })

  it('requires every term (AND)', () => {
    expect(filterNodes(g, 'cluster spark', EMPTY_FILTERS).map(n => n.id)).toEqual(['memory:memory:1'])
    expect(filterNodes(g, 'cluster nonexistent', EMPTY_FILTERS)).toEqual([])
  })
})

describe('effectiveRange (date modes)', () => {
  it('range mode collapses from/to ISO days to an inclusive unix window', () => {
    const { from, to } = effectiveRange({ ...EMPTY_FILTERS, from: '2026-01-01', to: '2026-01-01' })

    expect(from).toBe(new Date(2026, 0, 1).getTime() / 1000)
    // Inclusive of the whole day.
    expect(to).toBe(new Date(2026, 0, 1).getTime() / 1000 + 86_399)
  })

  it('year mode spans Jan 1 → Dec 31 of the year', () => {
    const { from, to } = effectiveRange({ ...EMPTY_FILTERS, dateMode: 'year', year: '2025' })

    expect(from).toBe(new Date(2025, 0, 1).getTime() / 1000)
    expect(to).toBe(new Date(2026, 0, 1).getTime() / 1000 - 1)
  })

  it('yearMonth mode spans a single month, inclusive of its last day', () => {
    const { from, to } = effectiveRange({ ...EMPTY_FILTERS, dateMode: 'yearMonth', month: '02', year: '2024' })

    // Feb 2024 (leap year) → through Feb 29.
    expect(from).toBe(new Date(2024, 1, 1).getTime() / 1000)
    expect(to).toBe(new Date(2024, 2, 1).getTime() / 1000 - 1)
  })

  it('widens (unbounded) when the selection is blank/partial', () => {
    expect(effectiveRange({ ...EMPTY_FILTERS, dateMode: 'year', year: '' })).toEqual({ from: null, to: null })
    // Year set, no month → the whole year, not excluded.
    const wholeYear = effectiveRange({ ...EMPTY_FILTERS, dateMode: 'yearMonth', month: '', year: '2025' })
    expect(wholeYear.from).toBe(new Date(2025, 0, 1).getTime() / 1000)
    expect(wholeYear.to).toBe(new Date(2026, 0, 1).getTime() / 1000 - 1)
  })

  it('filterNodes honors year mode', () => {
    const g2 = graph([
      node({ id: 'a', timestamp: new Date(2024, 5, 1).getTime() / 1000 }),
      node({ id: 'b', timestamp: new Date(2025, 5, 1).getTime() / 1000 })
    ])

    expect(filterNodes(g2, '', { ...EMPTY_FILTERS, dateMode: 'year', year: '2025' }).map(n => n.id)).toEqual(['b'])
  })
})

describe('conclusions (Honcho gate + derived level)', () => {
  // Honcho-derived inference → a conclusion. Honcho explicit statement and a
  // level-less honcho node → true memories. File memory → memory.
  const derived = node({ id: 'memory:honcho:0', memoryLevel: 'inductive', memorySource: 'honcho' })
  const derivedDeductive = node({ id: 'memory:honcho:1', memoryLevel: 'deductive', memorySource: 'honcho' })
  const explicit = node({ id: 'memory:honcho:2', memoryLevel: 'explicit', memorySource: 'honcho' })
  const levelless = node({ id: 'memory:honcho:3', memorySource: 'honcho' })
  const fileMem = node({ id: 'memory:memory:0', memorySource: 'memory' })
  const skill = node({ createdBy: 'agent', id: 'deploy', kind: 'skill', label: 'deploy' })

  it('isConclusion requires honcho provider, a honcho node, AND a derived level', () => {
    // Derived levels under honcho → conclusions.
    expect(isConclusion(derived, 'honcho')).toBe(true)
    expect(isConclusion(derivedDeductive, 'honcho')).toBe(true)
    // Explicit or missing level → a true memory, NOT a conclusion (the bug fix).
    expect(isConclusion(explicit, 'honcho')).toBe(false)
    expect(isConclusion(levelless, 'honcho')).toBe(false)
    // Provider not honcho → never a conclusion (never inferred from the node alone).
    expect(isConclusion(derived, 'chatgpt')).toBe(false)
    expect(isConclusion(derived, null)).toBe(false)
    // File memory under honcho → still not a conclusion.
    expect(isConclusion(fileMem, 'honcho')).toBe(false)
  })

  it('conclusionsEnabled only under honcho', () => {
    expect(conclusionsEnabled({ memoryProvider: 'honcho' })).toBe(true)
    expect(conclusionsEnabled({ memoryProvider: 'HONCHO' })).toBe(true)
    expect(conclusionsEnabled({ memoryProvider: 'chatgpt' })).toBe(false)
    expect(conclusionsEnabled({ memoryProvider: null })).toBe(false)
  })

  it("kind 'conclusion' matches only DERIVED honcho nodes; 'memory' keeps the true memories", () => {
    const g2 = graph([derived, explicit, levelless, fileMem, skill], [], 'honcho')

    // Only the derived-level honcho node is a conclusion.
    expect(filterNodes(g2, '', { ...EMPTY_FILTERS, kind: 'conclusion' }).map(n => n.id)).toEqual(['memory:honcho:0'])
    // 'memory' = memories that are NOT conclusions: the explicit + level-less
    // honcho nodes (true memories) AND the file memory.
    expect(filterNodes(g2, '', { ...EMPTY_FILTERS, kind: 'memory' }).map(n => n.id)).toEqual([
      'memory:honcho:2',
      'memory:honcho:3',
      'memory:memory:0'
    ])
    expect(filterNodes(g2, '', { ...EMPTY_FILTERS, kind: 'skill' }).map(n => n.id)).toEqual(['deploy'])
  })

  it('with no honcho provider, even a derived honcho node is an ordinary memory', () => {
    const g2 = graph([derived, fileMem], [], null)

    expect(filterNodes(g2, '', { ...EMPTY_FILTERS, kind: 'conclusion' })).toEqual([])
    expect(filterNodes(g2, '', { ...EMPTY_FILTERS, kind: 'memory' }).map(n => n.id)).toEqual([
      'memory:honcho:0',
      'memory:memory:0'
    ])
  })
})

describe('hasActiveNarrowing', () => {
  it('is false for the idle sidebar and true for any narrowing', () => {
    expect(hasActiveNarrowing('', EMPTY_FILTERS)).toBe(false)
    expect(hasActiveNarrowing('  ', EMPTY_FILTERS)).toBe(false)
    expect(hasActiveNarrowing('x', EMPTY_FILTERS)).toBe(true)
    expect(hasActiveNarrowing('', { ...EMPTY_FILTERS, kind: 'memory' })).toBe(true)
    expect(hasActiveNarrowing('', { ...EMPTY_FILTERS, kind: 'conclusion' })).toBe(true)
    expect(hasActiveNarrowing('', { ...EMPTY_FILTERS, source: 'chatgpt' })).toBe(true)
    expect(hasActiveNarrowing('', { ...EMPTY_FILTERS, from: '2026-01-01' })).toBe(true)
    expect(hasActiveNarrowing('', { ...EMPTY_FILTERS, dateMode: 'year', year: '2026' })).toBe(true)
    // A year mode with no year picked doesn't narrow (widens to unbounded).
    expect(hasActiveNarrowing('', { ...EMPTY_FILTERS, dateMode: 'year', year: '' })).toBe(false)
  })
})
