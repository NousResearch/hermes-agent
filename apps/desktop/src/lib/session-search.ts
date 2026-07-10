import type { SessionInfo } from '@/types/hermes'

import { sessionTitle } from './chat-runtime'
import { sessionSourceSearchTerms } from './session-source'

export function sessionMatchesSearch(session: SessionInfo, query: string): boolean {
  const needle = query.trim().toLowerCase()

  if (!needle) {
    return true
  }

  return [
    session.id,
    session._lineage_root_id ?? '',
    sessionTitle(session),
    session.preview ?? '',
    session.cwd ?? '',
    ...sessionSourceSearchTerms(session.source)
  ].some(value => value.toLowerCase().includes(needle))
}

/** Whether the query hits the session's assigned TITLE specifically.
 *  Titles are human-assigned intent (`/title` on any platform, desktop
 *  rename), so title matches rank above id/preview/cwd/source matches. */
export function sessionTitleMatches(session: SessionInfo, query: string): boolean {
  const needle = query.trim().toLowerCase()

  if (!needle) {
    return false
  }

  return (session.title ?? '').trim().toLowerCase().includes(needle)
}

/** Stable sort: title-matching sessions first, everything else after,
 *  preserving the existing relative order within each group. */
export function rankTitleMatchesFirst(sessions: SessionInfo[], query: string): SessionInfo[] {
  const needle = query.trim()

  if (!needle) {
    return sessions
  }

  const titleHits: SessionInfo[] = []
  const rest: SessionInfo[] = []

  for (const s of sessions) {
    if (sessionTitleMatches(s, needle)) {
      titleHits.push(s)
    } else {
      rest.push(s)
    }
  }

  return [...titleHits, ...rest]
}
