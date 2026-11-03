import type { NativeNotificationKind } from './native-notifications'
import type { SessionOwnerScope } from './session-request-router'

// Per-owner notification level: a plugin (Bot Mode) names how much a session's
// owner may notify — every kind, only what needs the user, or nothing. Core
// knows sessions and owners, not bots, so the plugin answers the question and
// core applies it. With no resolver, or none that knows the owner, every kind
// passes (the per-kind toggles alone decide, as before).
export type NotificationLevel = 'all' | 'needs-you' | 'off'

export interface NotificationLevelQuery {
  /** Connection-qualified owner when known; `connectionId` is null for a bare profile owner.
   *  `targetProfile` is the backend profile when a Desktop alias routes to a differently named one. */
  owner: null | { connectionId: null | string; profile: string; targetProfile?: string }
  runtimeSessionId: string
  storedSessionId: string
}

export type NotificationLevelResolver = (query: NotificationLevelQuery) => NotificationLevel | null | undefined

/** Kinds that need the user: a blocking prompt, or a turn that failed. */
const NEEDS_YOU_KINDS = new Set<NativeNotificationKind>(['approval', 'input', 'turnError'])

const resolvers = new Set<NotificationLevelResolver>()

export function registerNotificationLevelResolver(resolver: NotificationLevelResolver): () => void {
  resolvers.add(resolver)

  return () => {
    resolvers.delete(resolver)
  }
}

export function notificationLevelAllows(level: NotificationLevel | null | undefined, kind: NativeNotificationKind) {
  return level === 'off' ? false : level === 'needs-you' ? NEEDS_YOU_KINDS.has(kind) : true
}

function ownerOf(scope: SessionOwnerScope): NotificationLevelQuery['owner'] {
  if (!scope) {
    return null
  }

  return typeof scope === 'string'
    ? { connectionId: null, profile: scope }
    : {
        connectionId: scope.connectionId || null,
        profile: scope.profile || 'default',
        targetProfile: scope.targetProfile || undefined
      }
}

/** The level the registered resolvers give this session; the first that knows it wins. */
export function sessionNotificationLevel(
  query: Omit<NotificationLevelQuery, 'owner'> & { owner: SessionOwnerScope }
): NotificationLevel | null {
  if (!resolvers.size) {
    return null
  }

  const resolved = { ...query, owner: ownerOf(query.owner) }

  for (const resolver of resolvers) {
    const level = resolver(resolved)

    if (level) {
      return level
    }
  }

  return null
}
