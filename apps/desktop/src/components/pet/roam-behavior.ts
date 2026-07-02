/**
 * Pure decision helpers for the floating pet's wander — the "what to do & when"
 * layer, split out from the geometry (`roam-geometry.ts`) and the RAF/DOM loop
 * (`use-pet-roam.ts`) so the *rhythm* of the roam is tunable in one place and
 * unit testable (every function takes an injectable `rng`).
 *
 * The goal is a calm, believable critter rather than a fidgeting one. Two ideas
 * from ambient game-AI carry the weight (see GameAIPro ch.36 "Breathing Life
 * into Your Background Characters" + standard idle/wander state machines):
 *
 *  1. **Loaf, don't pace.** A background character that picks a new walk on
 *     every beat reads as nervous. Most decision beats just keep resting;
 *     movement is the exception, not the default (`REST_CHANCE`).
 *  2. **Memoryless dwell times.** Uniform pauses feel metronomic. An
 *     exponential dwell — the classic model for idle durations — gives mostly
 *     short rests with the occasional long loaf, so the cadence never reads as a
 *     fixed pattern (`dwellMs` / `PAUSE_DWELL`).
 */

import type { Ledge } from './roam-geometry'

export type Rng = () => number

/** What the pet does when a rest beat ends. */
export type RoamMove = 'rest' | 'stroll' | 'hop'

export interface DwellRange {
  /** Mean of the exponential draw — the "typical" rest length. */
  meanMs: number
  /** Floor, so a near-zero draw never produces a jittery micro-pause. */
  minMs: number
  /** Ceiling, so a fat-tail draw (or a throttled tab) can't freeze the pet. */
  maxMs: number
}

// Rest length between beats: mostly short loafs, the occasional long one.
export const PAUSE_DWELL: DwellRange = { maxMs: 13000, meanMs: 4200, minMs: 1500 }
// Most beats the pet just keeps loafing — a critter that re-walks every beat
// reads as nervous, not alive.
export const REST_CHANCE = 0.62
// When it *does* move, chance it hops to another ledge vs. strolling this one.
export const HOP_CHANCE = 0.2
// Strolls should cover ground, not shuffle: travel at least this fraction of the
// ledge (or this many px, whichever is larger), up to the room available.
const STROLL_MIN_FRACTION = 0.45
const STROLL_MIN_PX = 110
// Bias toward the roomier side so the pet crosses the app instead of pacing one
// spot; the long tail of the coin still lets it double back now and then.
const STROLL_TOWARD_ROOM = 0.85

/**
 * Exponential (memoryless) dwell time, clamped to `[minMs, maxMs]`. With rng→0
 * this returns `minMs`; with rng→1 it saturates at `maxMs`; in between it's
 * `-ln(u)·meanMs`, so short rests dominate and long loafs are rare but possible.
 */
export function dwellMs({ meanMs, minMs, maxMs }: DwellRange, rng: Rng = Math.random): number {
  const u = 1 - rng() // map [0,1) → (0,1] so the log stays finite

  return Math.min(maxMs, Math.max(minMs, -Math.log(u) * meanMs))
}

/**
 * Decide a beat: rest (the common case), or — when the pet is actually going to
 * move — hop to a reachable ledge if one exists and the dice say so, else stroll
 * the current ledge. `canHop` is false when no neighbouring surface overlaps, so
 * the pet never "hops" in place.
 */
export function chooseMove(canHop: boolean, rng: Rng = Math.random): RoamMove {
  if (rng() < REST_CHANCE) {
    return 'rest'
  }

  return canHop && rng() < HOP_CHANCE ? 'hop' : 'stroll'
}

/**
 * A stroll destination (absolute x) on `ledge` that actually goes somewhere:
 * lean toward the side with more room and guarantee a decent minimum travel, so
 * the pet crosses the app rather than shuffling in place.
 */
export function pickStrollTarget(ledge: Ledge, fromX: number, rng: Rng = Math.random): number {
  const span = ledge.right - ledge.left

  if (span <= 4) {
    return ledge.left
  }

  const roomLeft = fromX - ledge.left
  const roomRight = ledge.right - fromX
  // Usually head to the roomier side; the long tail of the coin doubles back.
  const goRight = (rng() < STROLL_TOWARD_ROOM) === (roomRight >= roomLeft)
  const room = Math.max(0, goRight ? roomRight : roomLeft)
  const minDist = Math.min(room, Math.max(span * STROLL_MIN_FRACTION, STROLL_MIN_PX))
  const dist = minDist + rng() * Math.max(0, room - minDist)

  return goRight ? fromX + dist : fromX - dist
}
