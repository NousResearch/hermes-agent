import { describe, expect, it } from 'vitest'

import { applyRepulsion, type Point } from './vault-forces'

function scatter(n: number, spread: number): Point[] {
  // Deterministic pseudo-random layout (no Math.random: the test must not flake).
  let seed = 42
  const next = () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296 - 0.5) * spread

  return Array.from({ length: n }, () => ({ vx: 0, vy: 0, x: next(), y: next() }))
}

describe('applyRepulsion', () => {
  it('gives the same forces with the grid as with checking every pair', () => {
    const grid = scatter(400, 2500)
    const brute = grid.map(p => ({ ...p }))

    applyRepulsion(grid, 1)
    applyRepulsion(brute, 1, 3200, true)

    grid.forEach((p, i) => {
      expect(p.vx).toBeCloseTo(brute[i]!.vx, 9)
      expect(p.vy).toBeCloseTo(brute[i]!.vy, 9)
    })
  })

  it('pushes two bodies apart equally and oppositely, and ignores ones out of range', () => {
    const [a, b, far] = [
      { vx: 0, vy: 0, x: 0, y: 0 },
      { vx: 0, vy: 0, x: 10, y: 0 },
      { vx: 0, vy: 0, x: 5000, y: 0 }
    ]

    applyRepulsion([a, b, far], 1)

    expect(a.vx).toBeLessThan(0)
    expect(b.vx).toBeCloseTo(-a.vx, 9)
    expect(far.vx).toBeCloseTo(0, 9)
  })
})
