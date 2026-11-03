/**
 * Which stored sessions are a bot's chat, for surfaces with no tile to carry
 * the scope — a Bot Chat normally lives in MAIN. The tile store re-homes tiles
 * on compression; this module re-homes the main tab's scope the same way.
 */

import { atom } from 'nanostores'

import { readJson, writeJson } from '@/lib/storage'

import type { SessionTileWorkspaceScope } from './session-states'

const BOT_CHAT_SCOPE_KEY = 'hermes.desktop.botChatSessions.v1'

/** Stored ids last opened as a bot's chat. A tile carries `workspaceMode`, but
 *  a bot chat normally lands in MAIN — `in-place` mints no tile when there is
 *  none to front — and main has no tile to carry the scope on. Kept here so a
 *  surface can still tell a companion chat from a working session, persisted
 *  so that survives a relaunch the way tile scope does. */
export const $botChatSessionIds = atom<ReadonlySet<string>>(
  new Set((readJson<unknown>(BOT_CHAT_SCOPE_KEY) as unknown[] | null)?.filter(id => typeof id === 'string') ?? [])
)

/** The bot-mode scope each stored id was last opened under, for the main tab
 *  (which has no tile to carry one). Window-local: the caption falls back to
 *  the stored title until the chat is opened again. */
export const $botChatScopes = atom<Readonly<Record<string, SessionTileWorkspaceScope>>>({})

export function rememberBotChatScope(storedSessionId: string, scope: SessionTileWorkspaceScope): void {
  const isBotChat = scope.workspaceMode === 'bots'
  const current = $botChatSessionIds.get()
  const { [storedSessionId]: previous, ...rest } = $botChatScopes.get()

  const changed = isBotChat
    ? previous?.workspaceOwnerKey !== scope.workspaceOwnerKey || previous?.workspaceTabTitle !== scope.workspaceTabTitle
    : Boolean(previous)

  if (changed) {
    $botChatScopes.set(isBotChat ? { ...rest, [storedSessionId]: scope } : rest)
  }

  if (current.has(storedSessionId) === isBotChat) {
    return
  }

  const next = new Set(current)

  if (isBotChat) {
    next.add(storedSessionId)
  } else {
    next.delete(storedSessionId)
  }

  $botChatSessionIds.set(next)
  writeJson(BOT_CHAT_SCOPE_KEY, next.size ? [...next] : null)
}

/** A compression rotation names the same Bot Chat by a new tip id; both the
 *  scope and the bot-chat mark follow it. Main has no tile to carry them, so
 *  without this the rotated tip read as a plain session: the main tab caption
 *  fell back to the stored "Bot Chat" and the composer brought the branch rail
 *  back. The previous id keeps its entry — route-follow re-selects the tip a
 *  beat after the rotation, and main still names the old id until then. */
export function carryBotChatScope(previousStoredSessionId: string, nextStoredSessionId: string): void {
  const scopes = $botChatScopes.get()
  const scope = scopes[previousStoredSessionId]

  if (scope && !scopes[nextStoredSessionId]) {
    $botChatScopes.set({ ...scopes, [nextStoredSessionId]: scope })
  }

  // The mark is persisted while the scopes are window-local: after a relaunch
  // only the mark is left to carry.
  const ids = $botChatSessionIds.get()

  if (ids.has(previousStoredSessionId) && !ids.has(nextStoredSessionId)) {
    const next = new Set(ids).add(nextStoredSessionId)

    $botChatSessionIds.set(next)
    writeJson(BOT_CHAT_SCOPE_KEY, [...next])
  }
}
