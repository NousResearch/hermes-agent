import { searchSessions } from '@/hermes'
import { normalize } from '@/lib/text'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { sessionLabel } from '../inline-refs'

import type { CompletionEntry } from './use-live-completion-adapter'

const MAX_ROWS = 12
// Full-text search scans message content: one-letter queries match nearly everything.
const SERVER_MIN_CHARS = 2

function sessionEntry(id: string, profile: string, title: string, meta: string): CompletionEntry {
  return {
    text: `@session:${profile}/${id}`,
    display: sessionLabel({ id, profile, title }),
    meta
  }
}

function matchesLocal(session: SessionInfo, q: string): boolean {
  return (
    !q ||
    session.id.toLowerCase().startsWith(q) ||
    normalize(session.title || '').includes(q) ||
    normalize(session.preview || '').includes(q)
  )
}

/** `@session:<text>` rows: loaded sidebar sessions of the active profile first (newest first,
 *  matched on title, last message or id prefix), then the server's full-text search so chats
 *  outside the loaded page are findable. Picking one inserts the same `@session:<profile>/<id>`
 *  chip a sidebar drag makes; the agent resolves it with session_search. */
export async function sessionMentionEntries(text: string, owner?: null | string): Promise<CompletionEntry[]> {
  const q = normalize(text)
  const profile = normalizeProfileKey(owner || $activeGatewayProfile.get())
  const seen = new Set<string>()
  const rows: CompletionEntry[] = []

  const local = $sessions
    .get()
    .filter((s: SessionInfo) => !s.archived && normalizeProfileKey(s.profile) === profile && matchesLocal(s, q))
    .sort((a: SessionInfo, b: SessionInfo) => (b.last_active || b.started_at || 0) - (a.last_active || a.started_at || 0))

  for (const session of local) {
    if (rows.length >= MAX_ROWS) {
      return rows
    }

    if (!seen.has(session.id)) {
      seen.add(session.id)
      rows.push(sessionEntry(session.id, profile, session.title || '', session.preview || ''))
    }
  }

  if (q.length < SERVER_MIN_CHARS) {
    return rows
  }

  try {
    const { results } = await searchSessions(text.trim(), profile)

    for (const hit of results) {
      if (rows.length >= MAX_ROWS) {
        break
      }

      if (!hit.archived && !seen.has(hit.session_id)) {
        seen.add(hit.session_id)
        // FTS highlight delimiters, as the sidebar strips them (sidebar/index.tsx stripFtsMarkers).
        const snippet = (hit.snippet || '').replaceAll('>>>', '').replaceAll('<<<', '')

        rows.push(sessionEntry(hit.session_id, profile, hit.title || '', snippet))
      }
    }
  } catch {
    // Search is best effort: the loaded sessions above still answer.
  }

  return rows
}
