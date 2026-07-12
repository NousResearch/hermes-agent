import { normalize } from '@/lib/text'
import type { SessionInfo } from '@/types/hermes'

import { sessionTitle } from './chat-runtime'
import { sessionSourceSearchTerms } from './session-source'

/** Tokenize a query for multi-term matching: lowercased whitespace split. */
function queryTokens(query: string): string[] {
  return normalize(query).split(/\s+/).filter(Boolean)
}

/** All searchable text fields of a session, lowercased once. */
function searchFields(session: SessionInfo) {
  return {
    channel: (session.display_name ?? '').toLowerCase(),
    cwd: (session.cwd ?? '').toLowerCase(),
    id: `${session.id} ${session._lineage_root_id ?? ''}`.toLowerCase(),
    platformTerms: sessionSourceSearchTerms(session.source).map(t => t.toLowerCase()),
    preview: (session.preview ?? '').toLowerCase(),
    title: sessionTitle(session).toLowerCase()
  }
}

/** Every query token must land somewhere: title, channel/thread path
 *  (display_name), platform name/alias, id, preview, or cwd. Multi-token
 *  queries like "voice assistant discord" thereby match a session whose
 *  channel is "#voice-assitant" (sic) on source discord — each token scores
 *  independently, so typos in channel names don't break phrase search. */
export function sessionMatchesSearch(session: SessionInfo, query: string): boolean {
  const tokens = queryTokens(query)

  if (tokens.length === 0) {
    return true
  }

  const f = searchFields(session)

  return tokens.every(
    tok =>
      f.title.includes(tok) ||
      f.channel.includes(tok) ||
      f.platformTerms.some(term => term.includes(tok)) ||
      f.id.includes(tok) ||
      f.preview.includes(tok) ||
      f.cwd.includes(tok)
  )
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

/** Whether the query hits the session's channel/thread path (display_name) —
 *  per-token, so "voice assistant" hits "Daemonarchy / #voice-assitant". */
export function sessionChannelMatches(session: SessionInfo, query: string): boolean {
  const tokens = queryTokens(query)
  const channel = (session.display_name ?? '').toLowerCase()

  if (tokens.length === 0 || !channel) {
    return false
  }

  return tokens.some(tok => channel.includes(tok))
}

/** Whether any query token names the session's platform (source/label/alias). */
export function sessionPlatformMatches(session: SessionInfo, query: string): boolean {
  const tokens = queryTokens(query)

  if (tokens.length === 0) {
    return false
  }

  const terms = sessionSourceSearchTerms(session.source).map(t => t.toLowerCase())

  return tokens.some(tok => terms.some(term => term === tok))
}

/** Stable sort into ranked groups: title hits first, then channel/thread-name
 *  hits (platform-named queries boost their platform's sessions within the
 *  group), then everything else — preserving relative order within groups.
 *  Grouping is per-token (mirroring the server-side search_sessions_by_title
 *  contract): "voice discord" title-ranks a session titled "Voice pipeline"
 *  even though the whole phrase appears in no single field. */
export function rankTitleMatchesFirst(sessions: SessionInfo[], query: string): SessionInfo[] {
  const needle = query.trim()
  const tokens = queryTokens(query)

  if (!needle || tokens.length === 0) {
    return sessions
  }

  const titleHits: SessionInfo[] = []
  const channelHits: SessionInfo[] = []
  const platformHits: SessionInfo[] = []
  const rest: SessionInfo[] = []

  for (const s of sessions) {
    const title = (s.title ?? '').trim().toLowerCase()

    if (title && tokens.some(tok => title.includes(tok))) {
      titleHits.push(s)
    } else if (sessionChannelMatches(s, needle)) {
      channelHits.push(s)
    } else if (sessionPlatformMatches(s, needle)) {
      platformHits.push(s)
    } else {
      rest.push(s)
    }
  }

  return [...titleHits, ...channelHits, ...platformHits, ...rest]
}
