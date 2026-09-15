import { normalize } from '@/lib/text'
import type { SessionInfo, SessionSearchResult } from '@/types/hermes'

import { sessionTitle } from './chat-runtime'
import { sessionSourceSearchTerms } from './session-source'

export function sessionMatchesSearch(session: SessionInfo, query: string): boolean {
  const needle = normalize(query)

  if (!needle) {
    return true
  }

  return [
    session.id,
    session._lineage_root_id ?? '',
    sessionTitle(session),
    session.preview ?? '',
    session.cwd ?? '',
    session.git_branch ?? '',
    ...sessionSourceSearchTerms(session.source)
  ].some(value => value.toLowerCase().includes(needle))
}

// The backend's FTS layer wraps matched terms in literal '>>>' / '<<<'
// highlight markers (sqlite snippet() delimiters — see hermes_state_search.py).
// Rows render as plain text, so the markers must be stripped or a search for
// "foo" paints ">>>foo<<<".
export function stripFtsMarkers(snippet: string): string {
  return snippet.replaceAll('>>>', '').replaceAll('<<<', '')
}

/** Synthesize a SessionInfo for a server hit that is not in the loaded store,
 *  so it renders in the same row component (resume works by id; the snippet
 *  stands in for the preview). `session_id` is the live compression tip — the
 *  documented resume identity — so `id` must be it, not the lineage root. */
export function searchResultToSession(result: SessionSearchResult): SessionInfo {
  const ts = result.started_at ?? result.session_started ?? Date.now() / 1000

  return {
    archived: result.archived ?? false,
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
    preview: stripFtsMarkers(result.snippet ?? '').trim() || result.preview || null,
    source: result.source ?? null,
    started_at: ts,
    title: result.title ?? null,
    tool_call_count: 0
  }
}

/** Merge instant client-side matches with server FTS hits, deduped by session
 *  id *and* compression lineage: local rows win, then each server hit whose
 *  lineage is not already present becomes the already-loaded row when we have
 *  one (`loadedById`, looked up by tip then lineage root), else a synthesized
 *  row. Lineage membership is tracked separately from `out`'s id keys — the two
 *  are different key spaces — so a hit that resolves to a conversation already
 *  listed under either identity is skipped, and rows keep the ids the list
 *  renders as React keys. Output order is local-then-server. */
export function mergeSessionSearchResults(
  localMatches: readonly SessionInfo[],
  serverMatches: readonly SessionSearchResult[],
  loadedById?: ReadonlyMap<string, SessionInfo>
): SessionInfo[] {
  const out = new Map<string, SessionInfo>()
  // Conversations already listed, keyed by compression lineage: the store can
  // hold the tip while the backend matched the lineage root (or the reverse),
  // so a hit for a conversation we already have must not become a second row.
  const seenLineages = new Set<string>()

  for (const session of localMatches) {
    out.set(session.id, session)

    if (session._lineage_root_id) {
      seenLineages.add(session._lineage_root_id)
    }
  }

  for (const match of serverMatches) {
    const root = match.lineage_root ?? null

    // A hit can also arrive keyed by a lineage root we already represent — the
    // tip/root pair can come tip-first or root-first — so test the hit's own id
    // against the seen lineages, not just its root.
    if (
      !match.session_id ||
      out.has(match.session_id) ||
      seenLineages.has(match.session_id) ||
      (root && (out.has(root) || seenLineages.has(root)))
    ) {
      continue
    }

    const loaded = loadedById?.get(match.session_id) ?? (root ? loadedById?.get(root) : undefined)

    if (!loaded) {
      out.set(match.session_id, searchResultToSession(match))

      if (root) {
        seenLineages.add(root)
      }

      continue
    }

    // A loaded row already listed under either of its identities is the same row.
    if (out.has(loaded.id) || (loaded._lineage_root_id && seenLineages.has(loaded._lineage_root_id))) {
      continue
    }

    out.set(loaded.id, loaded)

    if (loaded._lineage_root_id) {
      seenLineages.add(loaded._lineage_root_id)
    }

    if (root) {
      seenLineages.add(root)
    }
  }

  return [...out.values()]
}
