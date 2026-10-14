import { dropSessionState } from '@/store/session-states'

import type { ClientSessionState } from '../../../types'

/**
 * Take this session's warm runtime cache entry — only while it still BELONGS to
 * the session being resumed. A pooled profile backend that gets idle-reaped and
 * respawned (pruneSecondaryGateways) re-mints runtime ids, so a recycled id can
 * resolve to a live-but-DIFFERENT session's cache entry. The session.activate
 * 404 guard only catches a fully-DEAD id — a recycled-live id 200s, so an
 * unchecked hit paints the wrong transcript under the current route (the "open
 * chat A, chat B loads" bug). On a mismatch the mapping is cross-wired: purge
 * both sides and report a miss so the caller falls through to a full resume that
 * rebinds a correct id.
 */
export function takeWarmSessionCache(
  runtimeIdByStoredSessionId: Map<string, string>,
  sessionStateByRuntimeId: Map<string, ClientSessionState>,
  storedSessionId: string
): { runtimeId: string; state: ClientSessionState } | null {
  const runtimeId = runtimeIdByStoredSessionId.get(storedSessionId)
  const state = runtimeId ? sessionStateByRuntimeId.get(runtimeId) : undefined

  if (!runtimeId || !state) {
    return null
  }

  if (state.storedSessionId !== storedSessionId) {
    runtimeIdByStoredSessionId.delete(storedSessionId)
    sessionStateByRuntimeId.delete(runtimeId)
    dropSessionState(runtimeId)

    return null
  }

  return { runtimeId, state }
}
