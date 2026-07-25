import { normalize } from '@/lib/text'
import type { SessionInfo } from '@/types/hermes'

import { sessionTitle } from './chat-runtime'
import {
  bestMatch,
  rankFields,
  type FieldMatch,
  type FieldSpec,
  type RankedHit,
  type RankOptions
} from './search-match'
import { sessionSourceSearchTerms } from './session-source'

export function sessionSearchFields(session: SessionInfo): FieldSpec[] {
  return [
    { field: 'id', value: session.id },
    { field: 'id', value: session._lineage_root_id ?? '' },
    { field: 'title', value: session.title?.trim() || sessionTitle(session) },
    { field: 'preview', value: session.preview ?? '' },
    { field: 'cwd', value: session.cwd ?? '' },
    { field: 'branch', value: session.git_branch ?? '' },
    { field: 'source', value: sessionSourceSearchTerms(session.source).join(' ') }
  ]
}

export function rankSession(
  session: SessionInfo,
  query: string,
  opts?: RankOptions
): RankedHit<SessionInfo> | null {
  const needle = normalize(query)

  if (!needle) {
    return { item: session, score: 1, matches: [] }
  }

  const ranked = rankFields(sessionSearchFields(session), needle, opts)

  if (!ranked || ranked.score <= 0) {
    return null
  }

  return { item: session, score: ranked.score, matches: ranked.matches }
}

export function sessionMatchesSearch(session: SessionInfo, query: string): boolean {
  const needle = normalize(query)

  if (!needle) {
    return true
  }

  return rankSession(session, query) != null
}

export function sessionBestMatch(session: SessionInfo, query: string): FieldMatch | undefined {
  return bestMatch(rankSession(session, query)?.matches ?? [])
}
