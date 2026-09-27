import { computed, type ReadableAtom } from 'nanostores'

import type { SessionInfo } from '@/hermes'

import { $sidebarOrdering, type SidebarOrdering } from './layout'
import { $sessions } from './session'
import { $sessionDotStateById, type SessionDotState, sessionStatusRank } from './session-dot-state'
import { sessionCostUsd } from './sidebar-archive'

// Same array on every recompute, so the default (unranked) sidebar never churns
// its subscribers.
const UNRANKED: string[] = []

function rankBy(
  ordering: SidebarOrdering,
  dotStates: Record<string, SessionDotState>
): null | ((session: SessionInfo) => number) {
  switch (ordering) {
    case 'cost':
      return session => -sessionCostUsd(session)

    case 'created':
      return session => -session.started_at

    case 'status':
      return session => sessionStatusRank(dotStates[session.id])

    case 'tokens':
      return session => -(session.input_tokens + session.output_tokens)

    default:
      return null
  }
}

/**
 * The active sort key as a plain id order — the one ranking every sidebar
 * surface reads.
 *
 * The sort key used to be applied where the flat list is assembled, so it did
 * nothing at all once rows moved into groups: picking "cost" while grouped by
 * project or profile left every lane in the order the backend sent it. Ranking
 * lives here instead, above any one view, and each surface applies it to the
 * rows it owns — the flat list within its date dividers, a group within its
 * lane (and before it trims itself to a preview, so the rows it drops are the
 * ones the sort key ranked last).
 *
 * Empty for `updated` and `manual`: recency is the order sessions already
 * arrive in, and a hand-dragged sequence is the flat list's own business.
 */
export const $sidebarSessionRankIds: ReadableAtom<string[]> = computed(
  [$sidebarOrdering, $sessions, $sessionDotStateById],
  (ordering, sessions, dotStates) => {
    const rank = rankBy(ordering, dotStates)

    if (!rank) {
      return UNRANKED
    }

    return [...sessions].sort((a, b) => rank(a) - rank(b)).map(session => session.id)
  }
)
