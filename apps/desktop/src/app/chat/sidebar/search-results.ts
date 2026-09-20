import { sessionMatchesSearch } from '@/lib/session-search'
import type { SessionInfo, SessionSearchResult } from '@/types/hermes'

// FTS results cover sessions that aren't in the loaded page; synthesize a
// minimal SessionInfo so they render in the same row component (resume works
// by id; the snippet stands in for the preview).

// The backend's FTS layer wraps matched terms in literal '>>>' / '<<<'
// highlight markers (sqlite snippet() delimiters — see hermes_state_search.py).
// The sidebar renders the snippet as plain text, so the markers must be
// stripped or a search for "foo" paints rows titled ">>>foo<<<".
// Exported for tests.
export function stripFtsMarkers(snippet: string): string {
  return snippet.replaceAll('>>>', '').replaceAll('<<<', '')
}

export function searchResultToSession(result: SessionSearchResult): SessionInfo {
  const ts = result.session_started ?? Date.now() / 1000

  return {
    tags: result.tags ?? [],
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
    preview: stripFtsMarkers(result.snippet ?? '').trim() || null,
    source: result.source ?? null,
    started_at: ts,
    title: null,
    tool_call_count: 0
  }
}

export function mergeSessionSearchResults(
  query: string,
  sessions: SessionInfo[],
  serverMatches: SessionSearchResult[],
  sessionByAnyId: ReadonlyMap<string, SessionInfo>,
  matchesFilters: (session: SessionInfo) => boolean
): SessionInfo[] {
  if (!query) {return []}
  const out = new Map<string, SessionInfo>()

  for (const session of sessions) {
    if (sessionMatchesSearch(session, query)) {out.set(session.id, session)}
  }

  for (const match of serverMatches) {
    if (!out.has(match.session_id)) {
      out.set(match.session_id, sessionByAnyId.get(match.session_id) ?? searchResultToSession(match))
    }
  }

  return [...out.values()].filter(matchesFilters)
}
