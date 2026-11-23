import { readJson, writeJson } from '@/lib/storage'

import type { SessionOwnerRoute } from './session-request-router'
import type { SessionTileWorkspaceScope } from './session-states'

/** The bot-mode scope each STORED SESSION was last opened under, keyed by its
 *  stored id. A bot chat normally lives in MAIN, which has no tile to carry the
 *  scope, and the main tab captions a canonical Bot Chat by its bot only while
 *  it knows the owner — so the scope has to survive a relaunch the way tile
 *  scope does. Presentation and routing only: never Bot Chat identity, which
 *  is always re-resolved by name. */
const BOT_CHAT_SCOPE_KEY = 'hermes.desktop.botChatScopeByStoredSession.v2'

/** Superseded: the stored ids alone, with no owner to caption them by. */
const LEGACY_BOT_CHAT_SESSIONS_KEY = 'hermes.desktop.botChatSessions.v1'

type StoredBotChatScope = Pick<SessionTileWorkspaceScope, 'ownerRoute' | 'workspaceOwnerKey' | 'workspaceTabTitle'>

export interface BotChatScopeState {
  ids: ReadonlySet<string>
  scopes: Readonly<Record<string, SessionTileWorkspaceScope>>
}

function parseOwnerRoute(raw: unknown): SessionOwnerRoute | undefined {
  const route = raw as Partial<SessionOwnerRoute> | null | undefined

  return route && typeof route.connectionId === 'string' && typeof route.profile === 'string'
    ? {
        connectionId: route.connectionId,
        ...(route.mode === 'local' || route.mode === 'remote' ? { mode: route.mode } : {}),
        profile: route.profile,
        ...(typeof route.targetProfile === 'string' ? { targetProfile: route.targetProfile } : {})
      }
    : undefined
}

function parseScope(raw: unknown): SessionTileWorkspaceScope | undefined {
  const stored = raw as null | StoredBotChatScope | undefined

  if (!stored || typeof stored.workspaceOwnerKey !== 'string' || !stored.workspaceOwnerKey) {
    return undefined
  }

  const ownerRoute = parseOwnerRoute(stored.ownerRoute)

  return {
    workspaceMode: 'bots',
    workspaceOwnerKey: stored.workspaceOwnerKey,
    ...(typeof stored.workspaceTabTitle === 'string' ? { workspaceTabTitle: stored.workspaceTabTitle } : {}),
    ...(ownerRoute ? { ownerRoute } : {})
  }
}

/** The remembered bot chats, migrating the old list of ids: those stay marked
 *  as bot chats with no scope until they are opened again. */
export function loadBotChatScopes(): BotChatScopeState {
  const stored = readJson<unknown>(BOT_CHAT_SCOPE_KEY)

  if (stored && typeof stored === 'object' && !Array.isArray(stored)) {
    const scopes: Record<string, SessionTileWorkspaceScope> = {}

    for (const [id, raw] of Object.entries(stored as Record<string, unknown>)) {
      const scope = parseScope(raw)

      if (scope) {
        scopes[id] = scope
      }
    }

    return { ids: new Set(Object.keys(stored)), scopes }
  }

  const legacy = readJson<unknown>(LEGACY_BOT_CHAT_SESSIONS_KEY)

  return {
    ids: new Set(Array.isArray(legacy) ? legacy.filter((id): id is string => typeof id === 'string') : []),
    scopes: {}
  }
}

export function saveBotChatScopes({ ids, scopes }: BotChatScopeState): void {
  const stored: Record<string, StoredBotChatScope> = {}

  for (const id of ids) {
    const scope = scopes[id]

    stored[id] = scope
      ? {
          ...(scope.ownerRoute ? { ownerRoute: scope.ownerRoute } : {}),
          workspaceOwnerKey: scope.workspaceOwnerKey,
          ...(scope.workspaceTabTitle === undefined ? {} : { workspaceTabTitle: scope.workspaceTabTitle })
        }
      : {}
  }

  writeJson(BOT_CHAT_SCOPE_KEY, ids.size ? stored : null)
  // Always cleared: an emptied v2 key would otherwise fall back to the old list.
  writeJson(LEGACY_BOT_CHAT_SESSIONS_KEY, null)
}

function sameBotChatScope(a: SessionTileWorkspaceScope | undefined, b: SessionTileWorkspaceScope): boolean {
  return (
    a?.workspaceOwnerKey === b.workspaceOwnerKey &&
    a?.workspaceTabTitle === b.workspaceTabTitle &&
    a?.ownerRoute?.connectionId === b.ownerRoute?.connectionId &&
    a?.ownerRoute?.profile === b.ownerRoute?.profile &&
    a?.ownerRoute?.targetProfile === b.ownerRoute?.targetProfile
  )
}

/** Every open records the scope it used: a bot-mode open marks the stored id
 *  and keeps its scope, any other open clears both. Null when nothing changed,
 *  so neither atom (the id set has subscribers of its own) moves for nothing. */
export function withBotChatScope(
  { ids, scopes }: BotChatScopeState,
  storedSessionId: string,
  scope: SessionTileWorkspaceScope
): BotChatScopeState | null {
  const isBotChat = scope.workspaceMode === 'bots'
  const { [storedSessionId]: previous, ...rest } = scopes
  const scopeChanged = isBotChat ? !sameBotChatScope(previous, scope) : Boolean(previous)
  const marked = ids.has(storedSessionId) === isBotChat

  if (!scopeChanged && marked) {
    return null
  }

  const nextIds = new Set(ids)

  if (isBotChat) {
    nextIds.add(storedSessionId)
  } else {
    nextIds.delete(storedSessionId)
  }

  return {
    ids: marked ? ids : nextIds,
    scopes: scopeChanged ? (isBotChat ? { ...rest, [storedSessionId]: scope } : rest) : scopes
  }
}

/** The profile-keyed lifecycle, run beside the Bot tiles' own: a deleted
 *  profile's chats are forgotten (`keep` false), a renamed one's routes move
 *  (`rename`). Null when nothing changed. */
export function reownBotChatScopes(
  { ids, scopes }: BotChatScopeState,
  keep: (owner: SessionOwnerRoute | undefined) => boolean,
  rename: (owner: SessionOwnerRoute | undefined) => SessionOwnerRoute | undefined = owner => owner
): BotChatScopeState | null {
  const nextIds = new Set(ids)
  const nextScopes: Record<string, SessionTileWorkspaceScope> = {}
  let changed = false

  for (const [id, scope] of Object.entries(scopes)) {
    if (!keep(scope.ownerRoute)) {
      nextIds.delete(id)
      changed = true

      continue
    }

    const ownerRoute = rename(scope.ownerRoute)
    changed ||= ownerRoute !== scope.ownerRoute
    nextScopes[id] = ownerRoute === scope.ownerRoute ? scope : { ...scope, ownerRoute }
  }

  return changed ? { ids: nextIds, scopes: nextScopes } : null
}
