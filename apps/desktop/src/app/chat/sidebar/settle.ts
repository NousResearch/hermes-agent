import type { SessionInfo } from '@/hermes'
import { sessionMatchesStoredId } from '@/store/session'

/**
 * Where Settle lands when it settles the OPEN chat: the next recent below it,
 * else the one above, so a triage pass walks the list top to bottom without
 * a detour through an empty draft. Null means nothing is left to move to —
 * the caller falls back to the normal fresh draft.
 */
export function settleAdvanceTarget(sessions: readonly SessionInfo[], settledId: string): null | SessionInfo {
  const index = sessions.findIndex(session => sessionMatchesStoredId(session, settledId))

  if (index < 0) {
    return null
  }

  const candidate = (session: SessionInfo | undefined): session is SessionInfo =>
    Boolean(session && !session.archived && !sessionMatchesStoredId(session, settledId))

  return sessions.slice(index + 1).find(candidate) ?? sessions.slice(0, index).reverse().find(candidate) ?? null
}
