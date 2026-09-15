import type { ProfileScope } from '@/hermes'
import type { SessionOwnerRoute } from '@/store/session-request-router'

interface ModelPickerOwnerInput {
  ambientConnectionId?: string
  ambientProfile: string
  focusedStoredSessionId: null | string
  selectedStoredSessionId: null | string
  sessionTiles: readonly {
    ownerRoute?: SessionOwnerRoute
    storedSessionId: string
  }[]
}

export interface ModelPickerOwner {
  connectionId?: string
  profile: string
  route?: SessionOwnerRoute
}

export function resolveModelPickerProviderSetupScope(
  owner: ModelPickerOwner,
  ambientScope?: ProfileScope
): ProfileScope {
  if (!owner.route && ambientScope && typeof ambientScope === 'object') {
    const ambientProfile = (ambientScope.profile ?? '').trim() || 'default'
    const ambientConnectionId = (ambientScope.connectionId ?? '').trim() || undefined

    if (ambientProfile === owner.profile && ambientConnectionId === owner.connectionId) {
      return ambientScope
    }
  }

  return owner.connectionId ? { connectionId: owner.connectionId, profile: owner.profile } : owner.profile
}

/** Resolve one coherent owner for every picker operation. A tile route wins
 * only when the tile, rather than the primary session, owns keyboard focus. */
export function resolveModelPickerOwner({
  ambientConnectionId,
  ambientProfile,
  focusedStoredSessionId,
  selectedStoredSessionId,
  sessionTiles
}: ModelPickerOwnerInput): ModelPickerOwner {
  const route =
    focusedStoredSessionId && focusedStoredSessionId !== selectedStoredSessionId
      ? sessionTiles.find(tile => tile.storedSessionId === focusedStoredSessionId)?.ownerRoute
      : undefined

  return {
    connectionId: route?.connectionId || ambientConnectionId,
    profile: route?.targetProfile || route?.profile || ambientProfile,
    route
  }
}
