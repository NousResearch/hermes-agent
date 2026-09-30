import { computed, type ReadableAtom } from 'nanostores'

import type { SessionInfo } from '@/hermes'
import { stableArray } from '@/lib/stable-array'

import { $sidebarOrdering, type SidebarOrdering } from './layout'
import { $sessions } from './session'
import { $sessionDotStateById, type SessionDotState, sessionStatusBucket, sessionStatusRank } from './session-dot-state'
import { sessionCostUsd } from './sidebar-archive'

// Same array on every recompute, so the default (unranked) sidebar never churns
// its subscribers.
const UNRANKED: string[] = []

// How far the running tier sits above every idle row — bigger than any
// plausible recency spread (unix seconds), smaller than precision damage.
const RUNNING_TIER = 1e12

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

    case 'active':
      // #46560: sessions still doing work lead the list; everything else
      // falls back to the recency order it already had. "Running" is the
      // bucket a user names it by — working, stalled, or a background/
      // delegating process — NOT needs-input, whose turn ended waiting on
      // the user, and not unread. Recency (last_active || started_at)
      // decides within each tier, so a session finishing its turn sinks
      // back into the day's order with no restart.
      return session =>
        (sessionStatusBucket(dotStates[session.id]) === 'working' ? -RUNNING_TIER : 0) -
        (session.last_active || session.started_at || 0)

    case 'tokens':
      return session => -(session.input_tokens + session.output_tokens)

    default:
      return null
  }
}

// Same-array guarantee for the ranked path too: `active` recomputes on every
// status edge while a turn runs, and a recompute that produces the same order
// hands subscribers the array they already hold — the way UNRANKED does for
// the default view.
let rankedIds: readonly string[] = []

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

    const next = [...sessions].sort((a, b) => rank(a) - rank(b)).map(session => session.id)

    return (rankedIds = stableArray(rankedIds, next)) as string[]
  }
)
