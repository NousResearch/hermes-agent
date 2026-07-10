import type { SessionInfo } from '@/hermes'
import type { ProjectSessionSort } from '@/lib/project-session-sort'
import { flattenSessionsWithBranches, type SidebarSessionEntry } from '@/lib/session-branch-tree'

export type { ProjectSessionSort } from '@/lib/project-session-sort'

const sessionTitle = (session: SessionInfo): string => session.title?.trim() ?? ''

const sessionActivity = (session: SessionInfo): number => session.last_active || session.started_at || 0

const compareIds = (a: SessionInfo, b: SessionInfo): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)

function titleComparator(sort: ProjectSessionSort, locale: string): (a: SessionInfo, b: SessionInfo) => number {
  const collator = new Intl.Collator(locale, { numeric: true, sensitivity: 'base', usage: 'sort' })
  const direction = sort === 'title-asc' ? 1 : -1

  return (a, b) => {
    const aTitle = sessionTitle(a)
    const bTitle = sessionTitle(b)

    // Untitled rows remain after named sessions in both directions. Within that
    // fallback bucket, newest activity then stable ids keeps the result useful
    // and deterministic instead of moving every unnamed session arbitrarily.
    if (!aTitle || !bTitle) {
      if (Boolean(aTitle) !== Boolean(bTitle)) {
        return aTitle ? -1 : 1
      }

      return sessionActivity(b) - sessionActivity(a) || compareIds(a, b)
    }

    return collator.compare(aTitle, bTitle) * direction || sessionActivity(b) - sessionActivity(a) || compareIds(a, b)
  }
}

/**
 * Sort session rows inside the Projects tree without changing their membership.
 * Branch clusters remain parent-first and adjacent so pagination cannot orphan a
 * child row. `recent` intentionally preserves the backend's existing input order.
 */
export function sortProjectSessions(
  sessions: SessionInfo[],
  sort: ProjectSessionSort,
  locale = 'en'
): SessionInfo[] {
  if (sort === 'recent') {
    return sessions
  }

  return flattenSessionsWithBranches(sessions, titleComparator(sort, locale)).map(entry => entry.session)
}

/** Flatten rows for rendering while applying the selected project sort semantics. */
export function flattenProjectSessions(
  sessions: SessionInfo[],
  sort: ProjectSessionSort,
  locale = 'en'
): SidebarSessionEntry[] {
  return sort === 'recent'
    ? flattenSessionsWithBranches(sessions)
    : flattenSessionsWithBranches(sessions, titleComparator(sort, locale))
}
