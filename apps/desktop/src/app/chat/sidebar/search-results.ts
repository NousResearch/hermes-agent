// Sidebar search-result helpers, split out of index.tsx: the backend's FTS
// snippet markers, the row a server hit synthesizes into, and the merge of the
// client's own matches with the ranked server response.
import { type SessionInfo, type SessionSearchResult } from '@/hermes'
import { sessionMatchesSearch } from '@/lib/session-search'

// The backend's FTS layer wraps matched terms in literal '>>>' / '<<<'
// highlight markers (sqlite snippet() delimiters — see hermes_state_search.py).
// The sidebar renders the snippet as plain text, so the markers must be
// stripped or a search for "foo" paints rows titled ">>>foo<<<".
// Exported for tests.
export function stripFtsMarkers(snippet: string): string {
  return snippet.replaceAll('>>>', '').replaceAll('<<<', '')
}

// The backend already ships the real session title on every search hit
// (web_routers/sessions.py add_lineage_result enriches each result via
// get_session_rich_row). Map it onto the synthesized row so the sidebar
// paints the actual name; the snippet stays as the preview. Untitled
// sessions keep today's snippet fallback via sessionTitle().
// Exported for tests.
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
    last_active: result.last_active ?? ts,
    message_count: 0,
    model: result.model ?? null,
    output_tokens: 0,
    preview: stripFtsMarkers(result.snippet ?? '').trim() || null,
    source: result.source ?? null,
    started_at: ts,
    title: result.title?.trim() || null,
    tool_call_count: 0
  }
}

export function mergeSearchResults(
  sortedSessions: readonly SessionInfo[],
  query: string,
  serverMatches: readonly SessionSearchResult[],
  sessionByAnyId: ReadonlyMap<string, SessionInfo>,
  searchPending: boolean
): SessionInfo[] {
  if (!query) {
    return []
  }

  // While the request is in flight the client's own recency-ordered matches
  // are all there is — instant feedback while typing, and no leftovers from
  // whatever the previous query's request returned. Once the ranked server
  // response lands, it decides the order: the backend runs direct id matches
  // before FTS content hits, so pasting a session's exact id must keep that
  // hit on top instead of letting newer quoting sessions bury it.
  const out = new Map<string, SessionInfo>()

  if (searchPending) {
    for (const s of sortedSessions) {
      if (sessionMatchesSearch(s, query)) {
        out.set(s.id, s)
      }
    }

    return [...out.values()]
  }

  for (const match of serverMatches) {
    if (out.has(match.session_id)) {
      continue
    }

    const loaded = sessionByAnyId.get(match.session_id)
    out.set(match.session_id, loaded ?? searchResultToSession(match))
  }

  // Client-only matches that the server didn't return (e.g. cwd/git-branch
  // fields the FTS index doesn't cover) still deserve a row — after the
  // ranked hits, in recency order.
  for (const s of sortedSessions) {
    if (!out.has(s.id) && sessionMatchesSearch(s, query)) {
      out.set(s.id, s)
    }
  }

  return [...out.values()]
}
