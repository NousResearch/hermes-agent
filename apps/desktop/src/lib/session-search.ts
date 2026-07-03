import type { SessionSearchResult } from '@/hermes'
import type { SessionInfo } from '@/types/hermes'

import { sessionTitle } from './chat-runtime'
import { sessionSourceSearchTerms } from './session-source'

// FTS results cover sessions that aren't in the loaded page; synthesize a
// minimal SessionInfo so they render alongside loaded rows (resume works by
// id; the snippet stands in for the preview).
export function searchResultToSession(result: SessionSearchResult): SessionInfo {
  const ts = result.session_started ?? Date.now() / 1000

  return {
    archived: false,
    cwd: null,
    ended_at: null,
    id: result.session_id,
    _lineage_root_id: result.lineage_root ?? null,
    input_tokens: 0,
    is_active: false,
    last_active: ts,
    message_count: 0,
    model: result.model ?? null,
    output_tokens: 0,
    preview: result.snippet?.trim() || null,
    source: result.source ?? null,
    started_at: ts,
    title: null,
    tool_call_count: 0
  }
}

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
