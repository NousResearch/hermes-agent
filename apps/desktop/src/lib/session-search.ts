import { normalize } from '@/lib/text'
import type { SessionInfo } from '@/types/hermes'

import { sessionTitle } from './chat-runtime'
import { normalizeSessionSource, sessionSourceLabel, sessionSourceSearchTerms } from './session-source'

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

/** Per-session relevance score, mirroring the server-side
 *  search_sessions_by_title contract exactly:
 *
 *  - `matched` counts query tokens that hit a STRONG field (title,
 *    channel/thread path, platform name) — more matched tokens rank first,
 *    so for "desktop app" a session in the "Desktop App" thread (both
 *    tokens hit the channel path) beats a cron titled "skill-patch-applier"
 *    (only "app" substring-hits the title). This is the fix for the
 *    cron-junk-buries-channel-hits regression.
 *  - `tier` breaks ties by WHERE the best hit landed: whole-phrase title
 *    exact(0) > prefix(1) > substring(2) > token-in-title(3) >
 *    token-in-channel(4) > platform-only(5) > weak/none(6).
 */
function searchScore(session: SessionInfo, needle: string, tokens: string[]): [number, number] {
  const title = (session.title ?? '').trim().toLowerCase()
  const channel = (session.display_name ?? '').toLowerCase()
  const platformTerms = sessionSourceSearchTerms(session.source).map(t => t.toLowerCase())

  let matched = 0
  let bestTier = 6

  for (const tok of tokens) {
    let tier: number | null = null

    if (title && title.includes(tok)) {
      tier = 3
    } else if (channel && channel.includes(tok)) {
      tier = 4
    } else if (platformTerms.some(term => term === tok)) {
      tier = 5
    }

    if (tier !== null) {
      matched += 1
      bestTier = Math.min(bestTier, tier)
    }
  }

  // Whole-phrase title tiers preserve the historical contract that an
  // exact/prefix title hit beats everything else.
  if (title === needle) {
    bestTier = 0
  } else if (title.startsWith(needle)) {
    bestTier = 1
  } else if (needle && title.includes(needle)) {
    bestTier = 2
  }

  return [-matched, bestTier]
}

/** Stable relevance sort mirroring the server contract: sessions matching
 *  more query tokens in strong fields (title/channel/platform) first, ties
 *  broken by where the best hit landed, then by incoming order (recency). */
export function rankTitleMatchesFirst(sessions: SessionInfo[], query: string): SessionInfo[] {
  const needle = query.trim().toLowerCase()
  const tokens = queryTokens(query)

  if (!needle || tokens.length === 0) {
    return sessions
  }

  return sessions
    .map((session, index) => ({ index, score: searchScore(session, needle, tokens), session }))
    .sort((a, b) => {
      if (a.score[0] !== b.score[0]) {
        return a.score[0] - b.score[0]
      }

      if (a.score[1] !== b.score[1]) {
        return a.score[1] - b.score[1]
      }

      return a.index - b.index
    })
    .map(entry => entry.session)
}

/** Origin-context prefix for a search result row, per Ace's format:
 *
 *    Discord: voice-assitant: Desktop App     (guild dropped, '#' stripped)
 *    Telegram: Home
 *    TUI                                       (local surfaces: just the label)
 *
 *  Messaging sessions derive the path from display_name (the gateway's
 *  presentation path "Guild / #channel / Thread"); local surfaces (TUI,
 *  CLI, Desktop, Cron, …) reduce to the platform label alone. Returns null
 *  when there is nothing informative to show. */
export function sessionOriginContext(session: SessionInfo): null | string {
  const id = normalizeSessionSource(session.source)
  const label = sessionSourceLabel(id) ?? (id || null)

  if (!label) {
    return null
  }

  const dn = (session.display_name ?? '').trim()

  if (!dn) {
    return label
  }

  const parts = dn
    .split(' / ')
    .map(p => p.replace(/^#/, '').trim())
    .filter(Boolean)

  // Drop the server/guild segment when a channel path follows it — the
  // channel + thread are what identify the conversation ("Daemonarchy /
  // #voice-assitant / Desktop App" → "voice-assitant: Desktop App").
  const path = parts.length > 1 ? parts.slice(1) : parts

  return [label, ...path].join(': ')
}
