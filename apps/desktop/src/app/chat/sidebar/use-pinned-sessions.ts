import { useCallback, useMemo } from 'react'

import type { SessionInfo } from '@/types/hermes'

import { resolvePinnedSessions } from './session-index'

interface PinnedSessionsInput {
  // The opt-in "Sort sidebar alphabetically" preference: the section is A→Z by
  // title instead of in the hand-picked order.
  alphabetical: boolean
  allSessions: readonly SessionInfo[]
  pinnedSessionIds: readonly string[]
  sessionByAnyId: Map<string, SessionInfo>
  unconfirmedPinWrites: ReadonlySet<string>
}

/**
 * The Pinned section's rows plus the identity test every other list uses to
 * leave them out.
 *
 * Local pin ids first (hand-picked order), then server-flagged pins the local
 * set doesn't know about — a backend `pinned=1` row must never be invisible just
 * because localStorage is cold or was clobbered (#85969) — minus the rows whose
 * flag our own in-flight pin write already contradicts.
 */
export function usePinnedSessions({
  alphabetical,
  allSessions,
  pinnedSessionIds,
  sessionByAnyId,
  unconfirmedPinWrites
}: PinnedSessionsInput) {
  const pinnedSessions = useMemo(
    () => resolvePinnedSessions(pinnedSessionIds, sessionByAnyId, allSessions, unconfirmedPinWrites, alphabetical),
    [pinnedSessionIds, sessionByAnyId, allSessions, unconfirmedPinWrites, alphabetical]
  )

  // Every id a pin is reachable under: the raw stored ids, plus BOTH identities
  // of each session we resolved one to. A pin is stored on the durable lineage
  // root, but the lists that must filter it out are fed from three independent
  // fetches (recents, the messaging slice, the backend project tree) and each
  // can surface the same conversation under either its live tip or its root.
  // Comparing one identity against the other is how a pinned session ended up
  // rendered twice — once in Pinned, once in its project group.
  const pinnedIdentitySet = useMemo(() => {
    const ids = new Set(pinnedSessionIds)

    for (const session of pinnedSessions) {
      ids.add(session.id)

      if (session._lineage_root_id) {
        ids.add(session._lineage_root_id)
      }
    }

    return ids
  }, [pinnedSessionIds, pinnedSessions])

  // A pinned session belongs to the Pinned section and nowhere else, so every
  // other list filters it out. Match on either identity the row carries — a
  // backend snapshot can surface either side of a compression tip rotation.
  const isPinnedSession = useCallback(
    (session: SessionInfo) =>
      pinnedIdentitySet.has(session.id) ||
      (session._lineage_root_id != null && pinnedIdentitySet.has(session._lineage_root_id)),
    [pinnedIdentitySet]
  )

  return { isPinnedSession, pinnedSessions }
}
