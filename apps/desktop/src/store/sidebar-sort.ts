import { computed, type ReadableAtom } from 'nanostores'

import type { SessionInfo } from '@/hermes'

import { $sidebarOrdering, type PinnedSortKey, type SidebarOrdering } from './layout'
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

/** Recency as the sidebar reads it: last activity, else creation. Mirrored from
 *  the projects module rather than imported — that one lives under app/ and
 *  pulls the whole sidebar in behind it. */
const recency = (session: SessionInfo): number => session.last_active || session.started_at || 0

/** A row's displayed title, mirrored from lib/chat-runtime's `sessionTitle` —
 *  importing that module here would drag the chat runtime into the store. */
const displayedTitle = (session: SessionInfo): string =>
  session.title?.trim() || session.preview?.trim() || 'Untitled session'

/** Titles read the way a person scans a list: case and accents fold together,
 *  and embedded numbers sort naturally (10 after 2). Leaving the locale
 *  unspecified follows the app's own locale, so an accented alphabet orders the
 *  way its language does instead of by code point. */
const titleCollator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' })

export function compareSessionTitles(a: SessionInfo, b: SessionInfo): number {
  const left = displayedTitle(a)
  const right = displayedTitle(b)
  const folded = titleCollator.compare(left, right)

  if (folded !== 0) {
    return folded
  }

  // Titles that fold together (differing only in case or accent) must still
  // land in ONE order — a stable sort alone would keep whatever order the pin
  // list happened to be in, so two sessions could swap on every recompute.
  if (left !== right) {
    return left < right ? -1 : 1
  }

  return a.id === b.id ? 0 : a.id < b.id ? -1 : 1
}

/**
 * Order the Pinned section's rows. `manual` is Pinned's own pin order — the
 * sequence the user builds by dragging a pin, and what the section shipped
 * with — so it arrives unchanged, like the flat list's `manual`.
 *
 * The rest mirror the filter menu's sort keys, except that `updated` ranks here
 * while the flat list gets it for free from the backend's own recency order: a
 * pin list is in hand-picked order, so nothing else would put it back in time.
 *
 * Array.sort is stable, so ties keep the pin order rather than shuffling.
 */
export function sortPinnedSessions(
  sessions: readonly SessionInfo[],
  ordering: PinnedSortKey,
  dotStates: Record<string, SessionDotState>
): SessionInfo[] {
  if (ordering === 'name') {
    return [...sessions].sort(compareSessionTitles)
  }

  const rank: null | ((session: SessionInfo) => number) =
    ordering === 'updated'
      ? session => -recency(session)
      : rankBy(ordering as Exclude<PinnedSortKey, 'name'>, dotStates)

  return rank ? [...sessions].sort((a, b) => rank(a) - rank(b)) : [...sessions]
}
