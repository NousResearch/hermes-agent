import { describe, expect, it } from 'vitest'

import { chooseMove, dwellMs, type DwellRange, HOP_CHANCE, pickStrollTarget, REST_CHANCE, type Rng } from './roam-behavior'
import type { Ledge } from './roam-geometry'

// Deterministic rng that replays a fixed sequence (last value sticks).
const seq =
  (...vals: number[]): Rng =>
  () =>
    vals.shift() ?? vals[vals.length - 1] ?? 0

const RANGE: DwellRange = { maxMs: 13000, meanMs: 4000, minMs: 1500 }
const ledge = (left: number, right: number, y = 0): Ledge => ({ left, right, y })

describe('dwellMs', () => {
  it('clamps the degenerate draws to the floor and ceiling', () => {
    // rng→0 ⇒ u=1 ⇒ -ln(1)·mean = 0, raised to the floor.
    expect(dwellMs(RANGE, () => 0)).toBe(RANGE.minMs)
    // rng→~1 ⇒ u→0 ⇒ -ln(u) blows up, capped at the ceiling.
    expect(dwellMs(RANGE, () => 1 - 1e-9)).toBe(RANGE.maxMs)
  })

  it('returns the mean at the exponential median point', () => {
    // rng = 1 - 1/e ⇒ u = 1/e ⇒ -ln(u) = 1 ⇒ exactly the mean.
    expect(dwellMs(RANGE, () => 1 - 1 / Math.E)).toBeCloseTo(RANGE.meanMs, 6)
  })

  it('stays within [min, max] across the whole rng domain', () => {
    let state = 0.123456789

    for (let i = 0; i < 5000; i++) {
      state = (state * 9301 + 0.49297) % 1 // cheap deterministic walk
      const ms = dwellMs(RANGE, () => state)
      expect(ms).toBeGreaterThanOrEqual(RANGE.minMs)
      expect(ms).toBeLessThanOrEqual(RANGE.maxMs)
    }
  })
})

describe('chooseMove', () => {
  it('rests whenever the first draw lands under restChance — even where it could hop', () => {
    expect(chooseMove(true, seq(0))).toBe('rest')
    expect(chooseMove(false, seq(REST_CHANCE - 1e-9))).toBe('rest')
  })

  it('strolls when moving with nowhere to hop', () => {
    expect(chooseMove(false, seq(0.99))).toBe('stroll')
  })

  it('hops only when moving, a ledge is reachable, and the second draw says so', () => {
    expect(chooseMove(true, seq(0.99, HOP_CHANCE - 1e-9))).toBe('hop')
    expect(chooseMove(true, seq(0.99, HOP_CHANCE))).toBe('stroll')
  })

  it('treats restChance as a strict lower bound (boundary stays a move)', () => {
    expect(chooseMove(false, seq(REST_CHANCE))).toBe('stroll')
  })

  it('loafs far more than it roams over a long run (the whole point)', () => {
    let state = 0.314159
    const rng: Rng = () => (state = (state * 16807 + 0.5) % 1)
    let rests = 0
    const N = 20000

    for (let i = 0; i < N; i++) {
      if (chooseMove(true, rng) === 'rest') {
        rests++
      }
    }

    // ~62% rests; assert the contract (majority loafing), not the exact rate.
    expect(rests / N).toBeGreaterThan(0.5)
  })
})

describe('pickStrollTarget', () => {
  it('collapses to the left edge on a ledge too narrow to walk', () => {
    expect(pickStrollTarget(ledge(100, 102), 100, seq(0))).toBe(100)
  })

  it('lands inside the ledge and clears the minimum travel distance', () => {
    const wide = ledge(0, 1000)
    const from = 500
    const x = pickStrollTarget(wide, from, seq(0.5, 0))

    expect(x).toBeGreaterThanOrEqual(wide.left)
    expect(x).toBeLessThanOrEqual(wide.right)
    expect(Math.abs(x - from)).toBeGreaterThanOrEqual(110) // STROLL_MIN_PX
  })

  it('heads toward the side with more room', () => {
    // Pinned near the right wall, the roomier side is left. First draw clears the
    // rare double-back coin ⇒ it commits to the roomy (left) side ⇒ target < x.
    const x = pickStrollTarget(ledge(0, 1000), 950, seq(0.5, 0))
    expect(x).toBeLessThan(950)
  })
})
