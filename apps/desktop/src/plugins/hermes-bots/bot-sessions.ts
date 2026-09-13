/**
 * Per-bot session browser: every stored chat that lives in ONE bot's profile,
 * listed inside the Bots rail on request.
 *
 * This is a READ surface over `session.list`, not an identity. The canonical
 * Bot Chat is still resolved by NAME through the registry (see
 * canonical-chat.ts) — this browser never stores a session pointer, never picks
 * "the" bot chat for the row, and never decides where a row click lands. It
 * only lets the user see and reopen the side-chats (`+` threads, CLI sessions,
 * routine runs) a bot's profile accumulated, which the Sessions rail hides
 * behind the active-profile scope.
 */

import { atom, host } from '@hermes/plugin-sdk'

import { CANONICAL_CHAT_TITLE, PROFILE_SESSION_LIST_LIMIT } from './canonical-chat'
import { botRosterKey } from './data'
import {
  backendTargetProfile,
  botConnectionRoute,
  botWorkspaceOwnerKey,
  requestForBot,
  setBotsWorkspaceOwner
} from './routing'
import type { RosterRow } from './types'

/** One `session.list` row as the browser reads it. `hidden` / `last_active`
 *  arrive from a gateway that publishes them; an older gateway omits both and
 *  the browser degrades to started_at order with no canonical badge. */
export interface BotSessionRow {
  id: string
  resolved_id?: string
  title?: string
  preview?: string
  started_at?: number
  last_active?: number
  message_count?: number
  hidden?: boolean
  source?: string
}

/** Roster key of the bot whose sessions are expanded in the rail; one at a
 *  time — the rail is narrow and two open browsers would push the roster off
 *  screen. Window-local on purpose: this is where the user is looking, not a
 *  preference. */
export const $botSessionsOpen = atom<string | null>(null)

export function toggleBotSessions(bot: RosterRow): void {
  const key = botRosterKey(bot)
  $botSessionsOpen.set($botSessionsOpen.get() === key ? null : key)
}

export function closeBotSessions(): void {
  $botSessionsOpen.set(null)
}

export function isBotSessionsOpen(bot: RosterRow, open: string | null): boolean {
  return open !== null && open === botRosterKey(bot)
}

/** The bot's own profile sessions, most recent first. Hidden rows are
 *  included so the canonical Bot Chat shows up (badged) beside the side-chats;
 *  the deny-listed sources (kanban/tool workers) never reach `session.list`. */
export async function listBotSessions(bot: RosterRow): Promise<BotSessionRow[]> {
  const route = botConnectionRoute(bot)
  const fallback = String(bot?.name || '').trim() || 'default'

  const res = await requestForBot<{ sessions?: BotSessionRow[] }>(bot, 'session.list', {
    profile: backendTargetProfile(route, fallback),
    limit: PROFILE_SESSION_LIST_LIMIT,
    include_hidden: true
  })

  const rows = Array.isArray(res?.sessions) ? res.sessions.filter(row => row && row.id) : []

  return rows.sort((a, b) => sessionActivity(b) - sessionActivity(a))
}

export function sessionActivity(row: BotSessionRow): number {
  return Number(row.last_active) || Number(row.started_at) || 0
}

/** The registry row itself — surfaced so the browser can badge it, never so
 *  it can be stored. Title equality is the same test the registry uses. */
export function isCanonicalRow(row: BotSessionRow): boolean {
  return String(row.title || '').trim() === CANONICAL_CHAT_TITLE
}

/** Open one listed session in the bot's workspace. Same open contract the
 *  canonical path uses (owner key, `bots` workspace, in-place intent) so the
 *  tab lands beside the Bot Chat instead of in the Sessions strip; the browser
 *  hands over the stored id and nothing else. */
export async function openBotSession(bot: RosterRow, row: BotSessionRow): Promise<void> {
  if (typeof host.openSession !== 'function') {
    throw new Error('This Hermes Desktop version cannot open stored sessions')
  }

  const route = botConnectionRoute(bot)
  const ownerKey = botWorkspaceOwnerKey(bot)
  setBotsWorkspaceOwner(ownerKey, bot)

  await host.openSession(String(row.resolved_id || row.id), {
    ...(route ? { route } : {}),
    profile: String(bot?.name || '').trim() || 'default',
    intent: 'in-place',
    awaitHydration: true,
    expectHistory: (row.message_count ?? 0) > 0,
    forceResume: true,
    keepAllProfilesScope: true,
    workspaceMode: 'bots',
    workspaceOwnerKey: ownerKey,
    ...(isCanonicalRow(row) ? { tabTitle: CANONICAL_CHAT_TITLE } : {})
  })
}
