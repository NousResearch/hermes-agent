/**
 * EXPIRY BOOKKEEPING for session-states — settle grace + dead-state pruning.
 *
 * Split out of `session-states.ts` (ratchet offset: that file is over its
 * FILE_LINES cap and may only shrink). Dependency-free by design — only
 * `Date.now`, Maps, and a type-only state import — so the facade imports from
 * here with no cycle risk.
 */

import type { ClientSessionState } from '@/app/types'

// --- Settle grace: keeps a just-finished session in the sidebar merge set ---
const SESSION_SETTLE_GRACE_MS = 30 * 1000
const SETTLED_GRACE_CAP = 400
const settledExpiry = new Map<string, number>()

export function markSettled(storedId: string): void {
  settledExpiry.set(storedId, Date.now() + SESSION_SETTLE_GRACE_MS)

  // Prune-on-write backstop: the map is otherwise swept only when read
  // (getRecentlySettledSessionIds). If nothing reads it for a long while —
  // no list refresh — entries accumulate without bound.
  if (settledExpiry.size > SETTLED_GRACE_CAP) {
    const now = Date.now()

    for (const [id, expiry] of settledExpiry) {
      if (settledExpiry.size <= SETTLED_GRACE_CAP) {
        break
      }

      if (expiry <= now) {
        settledExpiry.delete(id)
      }
    }
  }
}

export function clearSettled(storedId: string): void {
  settledExpiry.delete(storedId)
}

/** Drop every settle-grace entry (gateway wipe). */
export function clearSettledGrace(): void {
  settledExpiry.clear()
}

/** Stored ids whose turn ended within the grace window. Prunes expired. */
export function getRecentlySettledSessionIds(now: number = Date.now()): string[] {
  const live: string[] = []

  for (const [id, expiry] of settledExpiry) {
    if (expiry > now) {
      live.push(id)
    } else {
      settledExpiry.delete(id)
    }
  }

  return live
}

// --- Dead lightweight-state pruning ----------------------------------------
// `releaseSessionTranscript` keeps the runtimeId key forever (with an emptied
// transcript) and eviction otherwise happens only via explicit drop/wipe — a
// day of tile churn parks hundreds of shells. The facade sweeps fully-idle,
// unreferenced lightweight entries oldest-first until under the cap. Working
// turns, attention dots, tiles, and the active runtime never qualify;
// settle-grace survival is ID-based (settledExpiry above), not entry-based,
// so the grace keep-set is unaffected.
export const LIGHTWEIGHT_STATES_CAP = 300

/**
 * Runtime ids safe to drop from a `$sessionStates` snapshot: fully idle
 * (no messages, not busy/waiting), unreferenced, oldest-first, stopping once
 * the snapshot is back under the cap. Pure — the facade deletes the keys.
 */
export function deadLightweightKeys(
  states: Record<string, ClientSessionState>,
  isReferenced: (runtimeId: string, storedSessionId: string | null) => boolean
): string[] {
  const keys = Object.keys(states)

  if (keys.length <= LIGHTWEIGHT_STATES_CAP) {
    return []
  }

  // Iteration order is insertion order (untouched keys keep their position
  // across spreads), so the front is the least-recently-added — sweep there.
  const dead: string[] = []

  for (const runtimeId of keys) {
    if (keys.length - dead.length <= LIGHTWEIGHT_STATES_CAP) {
      break
    }

    const state = states[runtimeId]

    if (
      state &&
      state.messages.length === 0 &&
      !state.busy &&
      !state.needsInput &&
      !state.awaitingResponse &&
      !isReferenced(runtimeId, state.storedSessionId)
    ) {
      dead.push(runtimeId)
    }
  }

  return dead
}

/**
 * The sweep applied: copy without the dead keys, or null when nothing
 * qualifies (common case — stays a single read, no store write).
 */
export function sweepDeadLightweightStates(
  states: Record<string, ClientSessionState>,
  isReferenced: (runtimeId: string, storedSessionId: string | null) => boolean
): Record<string, ClientSessionState> | null {
  const dead = deadLightweightKeys(states, isReferenced)

  if (dead.length === 0) {
    return null
  }

  const next = { ...states }

  for (const runtimeId of dead) {
    delete next[runtimeId]
  }

  return next
}
