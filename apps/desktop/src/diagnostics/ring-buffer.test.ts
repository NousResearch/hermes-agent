import { describe, expect, it } from 'vitest'

import { DiagnosticsRingBuffer, estimateEventBytes } from './ring-buffer'

interface Probe {
  t: number
  type: 'probe'
  n: number
}

const probe = (n: number, t = n): Probe => ({ n, t, type: 'probe' })

const generous = { maxAgeMs: 300_000, maxBytes: 10_000_000, maxEvents: 1_000_000 }

describe('DiagnosticsRingBuffer', () => {
  it('evicts the oldest events at the count cap instead of growing', () => {
    const buffer = new DiagnosticsRingBuffer<Probe>({ ...generous, maxEvents: 4 })

    for (let n = 0; n < 500; n += 1) {
      buffer.push(probe(n))
    }

    expect(buffer.size).toBe(4)
    expect(buffer.entries().map(e => e.n)).toEqual([496, 497, 498, 499])
    expect(buffer.droppedCount).toBe(496)
  })

  it('evicts at the byte cap and keeps the tracked byte total bounded', () => {
    const oneEvent = estimateEventBytes(probe(0))
    const buffer = new DiagnosticsRingBuffer<Probe>({ ...generous, maxBytes: oneEvent * 3 })

    for (let n = 0; n < 100; n += 1) {
      buffer.push(probe(n))
    }

    expect(buffer.size).toBe(3)
    expect(buffer.byteSize).toBeLessThanOrEqual(oneEvent * 3)
    expect(buffer.entries().map(e => e.n)).toEqual([97, 98, 99])
  })

  it('drops events older than the capture window relative to the newest', () => {
    const buffer = new DiagnosticsRingBuffer<Probe>({ ...generous, maxAgeMs: 300_000 })

    buffer.push(probe(1, 0))
    buffer.push(probe(2, 50_000))
    buffer.push(probe(3, 250_000))
    // 400s in: the first two are outside the 300s window, the third is not.
    buffer.push(probe(4, 400_000))

    expect(buffer.entries().map(e => e.n)).toEqual([3, 4])
  })

  it('reports a stable size after compaction churn', () => {
    const buffer = new DiagnosticsRingBuffer<Probe>({ ...generous, maxEvents: 8 })

    for (let n = 0; n < 10_000; n += 1) {
      buffer.push(probe(n))
      expect(buffer.size).toBeLessThanOrEqual(8)
    }

    expect(buffer.entries()).toHaveLength(8)
    expect(buffer.entries()[0].n).toBe(9_992)
  })

  it('clears back to empty', () => {
    const buffer = new DiagnosticsRingBuffer<Probe>(generous)
    buffer.push(probe(1))
    buffer.clear()

    expect(buffer.size).toBe(0)
    expect(buffer.byteSize).toBe(0)
    expect(buffer.entries()).toEqual([])
  })
})
