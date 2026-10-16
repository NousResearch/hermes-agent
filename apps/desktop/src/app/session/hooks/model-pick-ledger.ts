// Per-session ledger of in-flight model picks, shared by every surface that
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
const picksBySession = new Map<string, SessionPicks>()
// Newest pick per session while it is unsettled; a settled newest pick leaves no entry.
const latestPickBySession = new Map<string, number>()

/** Register a pick. `before` becomes the confirmed selection only when no other pick is in flight. */
export function beginModelPick(sessionId: string, before: ModelPickSnapshot, painted: ModelPickSnapshot): number {
  const token = ++pickSequence
  const picks = picksBySession.get(sessionId)

  latestPickBySession.set(sessionId, token)

  if (picks) {
    picks.pending.push({ painted, token })
    picks.shown = painted
  } else {
    picksBySession.set(sessionId, { confirmed: before, pending: [{ painted, token }], shown: painted })
  }

  return token
}

/** Whether a newer pick for the session was made after this one. Ask before settling it. */
export function isPickSuperseded(sessionId: string, token: number): boolean {
  return latestPickBySession.get(sessionId) !== token
}

/**
 * Settle a pick. An applied pick becomes the confirmed selection. Returns the
 * snapshot to repaint — the newest pick still in flight, else the confirmed
 * selection — when that differs from what the ledger last painted; null means
 * the current paint stays.
 */
export function settleModelPick(sessionId: string, token: number, applied: boolean): ModelPickSnapshot | null {
  const picks = picksBySession.get(sessionId)
  const index = picks?.pending.findIndex(pick => pick.token === token) ?? -1

  if (!picks || index < 0) {
    return null
  }

  const [settled] = picks.pending.splice(index, 1)

  if (latestPickBySession.get(sessionId) === token) {
    latestPickBySession.delete(sessionId)
  }

  if (applied) {
    picks.confirmed = settled.painted
  }

  if (picks.pending.length === 0) {
    picksBySession.delete(sessionId)
  }

  const target = picks.pending.at(-1)?.painted ?? picks.confirmed

  if (applied || target === picks.shown) {
    return null
  }

  picks.shown = target

  return target
}
