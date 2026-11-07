// Per-owner/session ledger of in-flight model picks, shared by every surface that
// mounts useModelControls. A pick's own pre-pick snapshot is only the
// OPTIMISTIC paint of whatever pick came before it, so rollback must target
// the last confirmed selection (or the newest pick still in flight), never
// that snapshot.

import type { ComposerModelSource } from '@/store/session'

export interface ModelPickSnapshot {
  model: string
  provider: string
  reasoningEffortWire: string
  source: ComposerModelSource
}

interface PendingPick {
  painted: ModelPickSnapshot
  token: number
}

interface SessionPicks {
  confirmed: ModelPickSnapshot
  pending: PendingPick[]
  shown: ModelPickSnapshot
}

let pickSequence = 0
const picksByOwner = new Map<string, SessionPicks>()
// Newest pick per owner/session while it is unsettled; a settled newest pick leaves no entry.
const latestPickByOwner = new Map<string, number>()

/** Register a pick. `before` becomes the confirmed selection only when no other pick is in flight. */
export function beginModelPick(owner: string, before: ModelPickSnapshot, painted: ModelPickSnapshot): number {
  const token = ++pickSequence
  const picks = picksByOwner.get(owner)

  latestPickByOwner.set(owner, token)

  if (picks) {
    picks.pending.push({ painted, token })
    picks.shown = painted
  } else {
    picksByOwner.set(owner, { confirmed: before, pending: [{ painted, token }], shown: painted })
  }

  return token
}

/** Whether a newer pick for the session was made after this one. Ask before settling it. */
export function isPickSuperseded(owner: string, token: number): boolean {
  return latestPickByOwner.get(owner) !== token
}

/**
 * Settle a pick. An applied pick becomes the confirmed selection. Returns the
 * snapshot to repaint — the newest pick still in flight, else the confirmed
 * selection — when that differs from what the ledger last painted; null means
 * the current paint stays.
 */
export function settleModelPick(owner: string, token: number, applied: boolean): ModelPickSnapshot | null {
  const picks = picksByOwner.get(owner)
  const index = picks?.pending.findIndex(pick => pick.token === token) ?? -1

  if (!picks || index < 0) {
    return null
  }

  const [settled] = picks.pending.splice(index, 1)

  if (latestPickByOwner.get(owner) === token) {
    latestPickByOwner.delete(owner)
  }

  if (applied) {
    picks.confirmed = settled.painted
  }

  if (picks.pending.length === 0) {
    picksByOwner.delete(owner)
  }

  const target = picks.pending.at(-1)?.painted ?? picks.confirmed

  if (applied || target === picks.shown) {
    return null
  }

  picks.shown = target

  return target
}
