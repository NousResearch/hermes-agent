/**
 * Which stored sessions are a bot's chat, for surfaces with no tile to carry
 * the scope — a Bot Chat normally lives in MAIN. The tile store re-homes tiles
 * on compression; this module re-homes the main tab's scope the same way.
 */

import { atom } from 'nanostores'

import { readJson, writeJson } from '@/lib/storage'

import type { SessionTileWorkspaceScope } from './session-states'

const BOT_CHAT_SCOPE_KEY = 'hermes.desktop.botChatSessions.v1'
const BOT_CHAT_SCOPES_KEY = 'hermes.desktop.botChatScopes.v1'

/** Every rotation adds the tip and keeps the previous id (see
 *  `carryBotChatScope`), so the persisted map is capped: oldest entries go. */
const MAX_PERSISTED_SCOPES = 200

/** Stored ids last opened as a bot's chat. A tile carries `workspaceMode`, but
 *  a bot chat normally lands in MAIN — `in-place` mints no tile when there is
 *  none to front — and main has no tile to carry the scope on. Kept here so a
 *  surface can still tell a companion chat from a working session, persisted
 *  so that survives a relaunch the way tile scope does. */
export const $botChatSessionIds = atom<ReadonlySet<string>>(
  new Set((readJson<unknown>(BOT_CHAT_SCOPE_KEY) as unknown[] | null)?.filter(id => typeof id === 'string') ?? [])
)

const optionalString = (value: unknown) => value === undefined || typeof value === 'string'

function isStoredBotScope(value: unknown): value is SessionTileWorkspaceScope {
  const scope = value as null | Partial<SessionTileWorkspaceScope>

  return Boolean(
    scope &&
      typeof scope === 'object' &&
      scope.workspaceMode === 'bots' &&
      optionalString(scope.workspaceOwnerKey) &&
      optionalString(scope.workspaceTabTitle) &&
      optionalString(scope.ownerProfile)
  )
}

function storedBotChatScopes(): Record<string, SessionTileWorkspaceScope> {
  const raw = readJson<unknown>(BOT_CHAT_SCOPES_KEY)

  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(raw).filter((entry): entry is [string, SessionTileWorkspaceScope] => isStoredBotScope(entry[1]))
  )
}

/** The bot-mode scope each stored id was last opened under, for the main tab
 *  (which has no tile to carry one). Persisted like tile scope: a window-local
 *  map left a Bot Chat restored into main captioned by its stored "Bot Chat"
 *  after every relaunch, until the chat was opened from its bot again. */
export const $botChatScopes = atom<Readonly<Record<string, SessionTileWorkspaceScope>>>(storedBotChatScopes())

function setBotChatScopes(next: Readonly<Record<string, SessionTileWorkspaceScope>>): void {
  const entries = Object.entries(next)
  const kept = entries.length > MAX_PERSISTED_SCOPES ? Object.fromEntries(entries.slice(-MAX_PERSISTED_SCOPES)) : next

  $botChatScopes.set(kept)
  writeJson(BOT_CHAT_SCOPES_KEY, Object.keys(kept).length ? kept : null)
}

export function rememberBotChatScope(storedSessionId: string, scope: SessionTileWorkspaceScope): void {
  const isBotChat = scope.workspaceMode === 'bots'
  const current = $botChatSessionIds.get()
  const { [storedSessionId]: previous, ...rest } = $botChatScopes.get()

  const changed = isBotChat
    ? previous?.workspaceOwnerKey !== scope.workspaceOwnerKey || previous?.workspaceTabTitle !== scope.workspaceTabTitle
    : Boolean(previous)

  if (changed) {
    setBotChatScopes(isBotChat ? { ...rest, [storedSessionId]: scope } : rest)
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
    setBotChatScopes({ ...scopes, [nextStoredSessionId]: scope })
  }

  // A mark persisted before the scopes were has no scope to carry: the mark
  // still follows on its own.
  const ids = $botChatSessionIds.get()

  if (ids.has(previousStoredSessionId) && !ids.has(nextStoredSessionId)) {
    const next = new Set(ids).add(nextStoredSessionId)

    $botChatSessionIds.set(next)
    writeJson(BOT_CHAT_SCOPE_KEY, [...next])
  }
}
