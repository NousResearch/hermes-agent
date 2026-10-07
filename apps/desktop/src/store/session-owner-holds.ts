import { registryBackendScopeKey } from '@hermes/shared'
import { atom } from 'nanostores'

import { normalizeProfileKey } from './profile'
import type { SessionOwnerScope } from './session-request-router'

// ── Owner hold across the create → foreground gap ───────────────────────────
// A routed session.create returns a stored id on the owner's socket, but the
// surface that will PIN that socket (the selected primary thread, or a tile)
// is published later and asynchronously: navigate → route effect →
// $selectedStoredSessionId, or openSessionTile → $sessionTiles. In that gap
// the entry has no active request, is not yet foreground-bound and, if the
// user switched source meanwhile, is not the active key either — so the
// live-work pruner or a refcount-0 lease release could close the socket that
// holds the just-minted runtime before the first prompt.submit. The hold
// names the owner in foregroundSessionScopes from the moment the create
// returns until the foreground publication takes over (the stored id becomes
// selected or tiled), the caller releases it (failed create / drift close),
// or a bounded TTL expires — nothing latches.
const SESSION_OWNER_HOLD_TTL_MS = 60_000

const sessionOwnerHolds = new Map<
  string,
  { owner: SessionOwnerScope; timer: ReturnType<typeof setTimeout>; until: number }
>()

export const $sessionOwnerHoldRevision = atom(0)

function bumpSessionOwnerHoldRevision(): void {
  $sessionOwnerHoldRevision.set($sessionOwnerHoldRevision.get() + 1)
}

function forgetSessionOwnerHold(storedSessionId: string, publish: boolean): boolean {
  const hold = sessionOwnerHolds.get(storedSessionId)

  if (!hold) {
    return false
  }

  clearTimeout(hold.timer)
  sessionOwnerHolds.delete(storedSessionId)

  if (publish) {
    bumpSessionOwnerHoldRevision()
  }

  return true
}

export function holdSessionOwnerUntilForeground(storedSessionId: string, owner: SessionOwnerScope): () => void {
  const id = storedSessionId.trim()

  if (!id || !owner) {
    return () => undefined
  }

  forgetSessionOwnerHold(id, false)
  const until = Date.now() + SESSION_OWNER_HOLD_TTL_MS
  const timer = setTimeout(() => releaseSessionOwnerHold(id), SESSION_OWNER_HOLD_TTL_MS)

  sessionOwnerHolds.set(id, { owner, timer, until })
  bumpSessionOwnerHoldRevision()

  return () => releaseSessionOwnerHold(id)
}

export function releaseSessionOwnerHold(storedSessionId: string): void {
  forgetSessionOwnerHold(storedSessionId.trim(), true)
}

/** @internal Tests. */
export function _resetSessionOwnerHoldsForTests(): void {
  const hadHolds = sessionOwnerHolds.size > 0

  for (const hold of sessionOwnerHolds.values()) {
    clearTimeout(hold.timer)
  }

  sessionOwnerHolds.clear()

  if (hadHolds) {
    bumpSessionOwnerHoldRevision()
  }
}

/** Retire every hold whose scope `known` already covers, plus every expired
 *  one, and return the scopes the surviving holds still name. Called by
 *  foregroundSessionScopes (session-states.ts) as its create → foreground
 *  rung; a hold already covered by the publication (or past its TTL) retires
 *  without a recursive revision publish. */
export function sweepSessionOwnerHolds(known: Set<string>): Set<string> {
  const scopes = new Set<string>()
  const now = Date.now()

  for (const [storedSessionId, hold] of [...sessionOwnerHolds]) {
    const scope = holdScope(hold.owner)

    if (!scope || hold.until <= now || known.has(scope)) {
      // This recompute was already triggered by the covering publication (or
      // is itself observing expiry), so avoid recursively publishing.
      forgetSessionOwnerHold(storedSessionId, false)

      continue
    }

    scopes.add(scope)
  }

  return scopes
}

/** The backend scope key a hold's owner names, when it names one. */
export function holdScope(owner: SessionOwnerScope): string | null {
  if (typeof owner === 'string') {
    return normalizeProfileKey(owner) || null
  }

  const connectionId = owner?.connectionId?.trim()

  return connectionId ? registryBackendScopeKey(connectionId, normalizeProfileKey(owner?.profile ?? '')) : null
}
