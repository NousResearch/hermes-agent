import { describe, expect, it } from 'vitest'

import type { TraceSpanNode } from '@/store/trace'

import { buildTimeMap, niceTicks } from './time-map'

function node(kind: TraceSpanNode['kind'], start: number, end: number): TraceSpanNode {
  return {
    id: `${kind}:${start}`,
    parentId: null,
    name: kind,
    kind,
    start,
    end,
    duration: end - start,
    status: 'ok',
    sessionId: null,
    attributes: {},
    depth: 0,
    children: []
  }
}

describe('buildTimeMap', () => {
  it('maps busy time 1:1 and compresses idle gaps', () => {
    // Two 1s busy spans separated by a 100s idle gap.
    const nodes = [node('TOOL', 0, 1), node('TOOL', 101, 102)]
    const map = buildTimeMap(nodes, 0, 102)

    // Endpoints anchor.
    expect(map.toV(0)).toBe(0)
    expect(map.toReal(0)).toBe(0)
    expect(map.toReal(map.totalV)).toBeCloseTo(102)

    // The 100s gap is collapsed: virtual width ≪ real width.
    expect(map.totalV).toBeLessThan(20)
    expect(map.gaps).toHaveLength(1)
    expect(map.gaps[0]!.r1 - map.gaps[0]!.r0).toBeCloseTo(100)
  })

  it('round-trips real↔virtual within busy regions', () => {
    const map = buildTimeMap([node('LLM', 10, 20)], 10, 20)

    expect(map.toReal(map.toV(15))).toBeCloseTo(15)
  })

  it('survives degenerate input (only container spans)', () => {
    const map = buildTimeMap([node('AGENT', 0, 0)], 0, 0)

    expect(map.totalV).toBeGreaterThan(0)
    expect(Number.isFinite(map.toReal(0))).toBe(true)
  })
})

describe('niceTicks', () => {
  it('returns ascending ticks covering the range', () => {
    const ticks = niceTicks(0, 10)

    expect(ticks.length).toBeGreaterThan(1)
    expect(ticks).toEqual([...ticks].sort((a, b) => a - b))
    expect(ticks.at(-1)!).toBeLessThanOrEqual(10)
  })

  it('degenerates safely on a zero-width range', () => {
    expect(niceTicks(5, 5)).toEqual([5])
  })
})
