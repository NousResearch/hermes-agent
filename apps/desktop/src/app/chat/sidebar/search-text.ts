/**
 * Turning a server search hit into a sidebar row.
 *
 * Split out of `index.tsx` (a facade already at its length cap): pure string/row mapping,
 * no React, and the FTS-marker quirk lives with the test that pins it.
 */

import type { SessionInfo, SessionSearchResult } from '@/hermes'

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
