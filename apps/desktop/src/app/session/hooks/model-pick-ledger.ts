// Per-session ledger of in-flight model picks, shared by every surface that
// mounts useModelControls. A pick's own pre-pick snapshot is only the
// OPTIMISTIC paint of whatever pick came before it, so rollback must target
// the last confirmed selection (or the newest pick still in flight), never
// that snapshot.

import type { ComposerModelSelection } from '@/store/session'

export interface ModelPickSnapshot {
  composerSelection: ComposerModelSelection
  model: string
  provider: string
  reasoningEffortWire: string
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
// The primary pick each session queued for its next turn: the one saved as the next chat's launch pick, so
// only its refusal may release that (deferred-pick.ts). Every settled pick rewrites it: a tile's pick, or one
// that applied at once, retires it.
const queuedLaunchPicks = new Map<string, string>()

export function noteQueuedLaunchPick(sessionId: string, model: null | string): void {
  if (model === null) {
    queuedLaunchPicks.delete(sessionId)
  } else {
    queuedLaunchPicks.set(sessionId, model)
  }
}

/** Whether `model` is the session's queued launch pick; a refused pick is spent either way. */
export function takeQueuedLaunchPick(sessionId: string, model: unknown): boolean {
  if (queuedLaunchPicks.get(sessionId) !== model) {
    return false
  }

  queuedLaunchPicks.delete(sessionId)

  return true
}

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
