/**
 * Pixel pets on the HUD — the walk. Two sprites patrol the strip of headroom
 * above the bar, turn at the edges, and stop to look at the pointer when it
 * comes close. Pure so the behaviour is tested without a DOM or a clock.
 */

export interface PetWalkState {
  /** Left edge of the sprite, CSS px from the strip's left. */
  x: number
  /** +1 walks right, -1 walks left. */
  dir: -1 | 1
  /** Seconds left in the current idle pause (0 = walking). */
  idle: number
  /** Seconds of walking accumulated, drives the bob. */
  walked: number
  /** Seconds until the next random decision (turn / pause). */
  decide: number
  /** True while the pointer is close and the pet has stopped to look. */
  looking: boolean
}

/** How the pet moves right now: `patrol` wanders, `pace` hurries back and
 *  forth (tools running), `stand` stays put (thinking, waiting, done, failed). */
export type PetWalkMode = 'pace' | 'patrol' | 'stand'

export interface PetWalkInput {
  /** Width of the strip the pet can walk on, CSS px. */
  stripWidth: number
  /** Sprite width, CSS px. */
  width: number
  /** Pointer x in the same coordinates, or null when it is not over the HUD. */
  pointerX: number | null
  /** Walking speed, px/s. */
  speed: number
  /** Look at the pointer when it is within this many px of the sprite's centre. */
  lookRadius: number
  /** Deterministic randomness for turns and pauses (0..1). */
  random: () => number
  mode?: PetWalkMode
}

export const PET_WALK_SPEED = 26
export const PET_LOOK_RADIUS = 96

export function initialPetWalk(x: number, dir: -1 | 1 = 1): PetWalkState {
  return { x, dir, idle: 0, walked: 0, decide: 2 + 2 * Math.random(), looking: false }
}

export function stepPetWalk(state: PetWalkState, dt: number, input: PetWalkInput): PetWalkState {
  const { stripWidth, width, pointerX, speed, lookRadius, random, mode = 'patrol' } = input
  const maxX = Math.max(0, stripWidth - width)
  const centre = state.x + width / 2

  // Someone is here: stop, face them, and stay put while they hang around.
  if (pointerX !== null && Math.abs(pointerX - centre) <= lookRadius) {
    const dir: -1 | 1 = pointerX >= centre ? 1 : -1

    return { ...state, dir, looking: true, idle: 0 }
  }

  let { x, dir, idle, walked, decide } = state

  // Standing: nothing moves, but a pause that was running keeps counting so
  // the pet does not resume mid-step the moment the turn ends.
  if (mode === 'stand') {
    return { x, dir, idle: Math.max(0, idle - dt), walked, decide, looking: false }
  }

  if (state.looking) {
    // They left: pause a beat before wandering off, like a cat losing interest.
    return { ...state, looking: false, idle: 0.6 + random() * 0.8 }
  }

  if (idle > 0 && mode !== 'pace') {
    idle = Math.max(0, idle - dt)

    return { x, dir, idle, walked, decide, looking: false }
  }

  decide -= dt

  if (decide <= 0) {
    // Pacing: quick, restless turns and no pauses. Patrolling: the odd pause.
    decide = mode === 'pace' ? 0.6 + random() * 1.2 : 2 + random() * 4
    const roll = random()

    if (mode !== 'pace' && roll < 0.35) {
      idle = 0.8 + random() * 1.6

      return { x, dir, idle, walked, decide, looking: false }
    }

    if (roll < (mode === 'pace' ? 0.7 : 0.55)) {
      dir = dir === 1 ? -1 : 1
    }
  }

  const pace = mode === 'pace' ? 2.6 : 1
  x += dir * speed * pace * dt
  walked += dt * pace

  if (x <= 0) {
    x = 0
    dir = 1
  } else if (x >= maxX) {
    x = maxX
    dir = -1
  }

  return { x, dir, idle: 0, walked, decide, looking: false }
}

/** Vertical bob while walking: a 6 Hz two-step, 2 px tall, in CSS px. */
export function petBob(state: PetWalkState): number {
  if (state.idle > 0 || state.looking) {
    return 0
  }

  return Math.abs(Math.sin(state.walked * Math.PI * 3)) * 2
}
