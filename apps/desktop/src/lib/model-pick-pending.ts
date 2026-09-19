/**
 * v16 — a model pick whose RPC failed/hung is a PENDING selection, not a
 * failed one: the backend may still apply the switch (transport dropped the
 * response), and the authoritative `session.info` will report the real pair.
 *
 * Registry shape: map runtimeId → {model, provider, seq, at}. The state-cache
 * reconciles against incoming `session.info` (same pair → commit: paint +
 * sticky; different pair or expiry → drop without touching the sticky).
 *
 * TTL exists so a backend that genuinely REFUSED can never leave the pending
 * hanging forever: without a confirming `session.info` within the window the
 * entry dies and the sticky keeps the previous value.
 */

import { DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS } from '@/api/client'

export interface PendingModelPick {
  model: string
  provider: string
  seq: number
  at: number
}


const pending = new Map<string, PendingModelPick>()
let seqCounter = 0

/** The PRIMARY pane's runtime id — the ONLY authority that may repaint the
 * primary composer atoms. A late `session.info` naming the primary's runtime
 * may repaint them; a TILE's confirmation (even a focused tile the user is
 * typing in) may not — the tile's own slice and the cross-scope sticky are
 * updated unconditionally by the reconciler, and the primary composer is
 * deliberately left alone (the pick on a tile is not a primary pick). */
let primaryRuntimeProvider: null | (() => null | string) = null

export function setPrimaryRuntimeProvider(fn: null | (() => null | string)): void {
  primaryRuntimeProvider = fn
}

export function isPrimaryRuntime(runtimeId: string): boolean {
  return primaryRuntimeProvider?.() === runtimeId
}

export function registerPendingModelPick(runtimeId: string, model: string, provider: string): number {
  const seq = ++seqCounter
  pending.set(runtimeId, { model, provider, seq, at: Date.now() })

  return seq
}

export function consumePendingModelPick(runtimeId: string): PendingModelPick | undefined {
  return pending.get(runtimeId)
}

export function resolvePendingModelPick(runtimeId: string): PendingModelPick | undefined {
  const entry = pending.get(runtimeId)
  pending.delete(runtimeId)

  return entry
}

export function clearPendingModelPick(runtimeId: string): void {
  pending.delete(runtimeId)
}

/**
 * v16b catch-guard: check whether the pending is still the SAME pick (seq
 * match) WITHOUT consuming it. The catch path must not delete the pending —
 * a transport failure does not mean the switch failed, and the reconciling
 * session.info still needs the entry to commit composer + sticky. This only
 * answers whether a late RPC failure is allowed to roll the optimistic paint
 * back: false means a newer pick replaced this one, or the reconciler already
 * consumed it (session.info confirmed) — the confirmed state wins.
 */
export function pendingModelPickSeqMatches(runtimeId: string, seq: number): boolean {
  const entry = pending.get(runtimeId)

  return Boolean(entry && entry.seq === seq)
}

/**
 * v16b success-guard: remove the pending ONLY if it is still the SAME pick
 * (seq match). Returns true when the entry existed and was removed. A false
 * return means a newer pick replaced this one — a late success must not
 * delete the NEWER pick's reconciliation window.
 */
export function clearPendingModelPickIfSeqMatches(runtimeId: string, seq: number): boolean {
  const entry = pending.get(runtimeId)

  if (!entry || entry.seq !== seq) {
    return false
  }

  pending.delete(runtimeId)

  return true
}

/** The reconciler decides with this: a pending older than the window died
 *  without ever being confirmed (backend refused / no session.info ever
 *  arrived) — drop it, keep the sticky as-is. */
/**
 * v16b: derived from the RPC timeout, not hand-picked. The reconciler's window
 * must OUTLIVE the `config.set` request: a pick whose RPC times out at T+30s
 * enters the catch with the pending still fresh, so a `session.info` arriving
 * right after the failure still reconciles. Equal values (30s == 30s) left a
 * zero-width post-catch window.
 */
export const PENDING_MODEL_PICK_TTL_MS = DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS + 15_000

export function isPendingFresh(entry: PendingModelPick, now = Date.now()): boolean {
  return now - entry.at < PENDING_MODEL_PICK_TTL_MS
}

export function resetPendingModelPicksForTests(): void {
  pending.clear()
  seqCounter = 0
  primaryRuntimeProvider = null
}
