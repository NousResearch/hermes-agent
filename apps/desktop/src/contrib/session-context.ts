import { useStore } from '@nanostores/react'

import {
  $activeSessionId,
  $connection,
  $cronSessions,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  getSessionOwnerHints,
  ownerLookupSessionRows,
  sessionMatchesStoredId
} from '@/store/session'
import { sessionOwnerRouteFromRow } from '@/store/session-request-router'
import {
  $sessionStates,
  $sessionTiles,
  knownOwnerForSession,
  storedSessionIdForRuntimeId
} from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import type { PluginSessionContext } from './session'

export interface SessionContributionIdentity {
  runtimeSessionId?: string | null
  storedSessionId?: string | null
  row?: SessionInfo | null
}

/** Use the existing owner ladder, but never choose the first colliding list row. */
export function resolveSessionContributionContext(input: SessionContributionIdentity): PluginSessionContext | null {
  const storedSessionId =
    input.storedSessionId ??
    input.row?.id ??
    (input.runtimeSessionId ? storedSessionIdForRuntimeId(input.runtimeSessionId) : null) ??
    (input.runtimeSessionId === $activeSessionId.get() ? $selectedStoredSessionId.get() : null)

  const id = storedSessionId || input.runtimeSessionId

  if (!id) {
    return null
  }

  const tile = $sessionTiles.get().find(t => t.storedSessionId === storedSessionId)
  const explicitOwner = sessionOwnerRouteFromRow(input.row) ?? tile?.ownerRoute

  const owners = new Set([
    ...getSessionOwnerHints(id).map(o => JSON.stringify([o.connectionId, o.profile])),
    ...ownerLookupSessionRows()
      .filter(s => sessionMatchesStoredId(s, id) && s.connection_id)
      .map(s => JSON.stringify([s.connection_id, s.profile || 'default']))
  ])

  if (!explicitOwner && owners.size > 1) {
    return null
  }

  const owner = explicitOwner ?? knownOwnerForSession(id)

  if (!owner) {
    return null
  }

  const connection = $connection.get()

  const connectionId =
    typeof owner === 'string'
      ? connection?.connectionId || (connection?.mode === 'local' ? 'local' : '')
      : owner.connectionId

  const profile = typeof owner === 'string' ? owner : owner.profile

  if (!connectionId || !profile) {
    return null
  }

  // A row may share a durable id with another backend's open tile. Never
  // attach that tile's runtime to the explicitly qualified row.
  const tileMatches =
    !explicitOwner ||
    !tile?.ownerRoute ||
    (tile.ownerRoute.connectionId === connectionId && tile.ownerRoute.profile === profile)

  const primaryMatches = storedSessionId === $selectedStoredSessionId.get() && (!explicitOwner || owners.size <= 1)

  const runtimeSessionId =
    input.runtimeSessionId ??
    (tileMatches ? tile?.runtimeId : null) ??
    (primaryMatches ? $activeSessionId.get() : null) ??
    null

  return { runtimeSessionId, storedSessionId, profile, connectionId }
}

export function useSessionContributionContext(input: SessionContributionIdentity): PluginSessionContext | null {
  useStore($sessions)
  useStore($cronSessions)
  useStore($messagingSessions)
  useStore($sessionTiles)
  useStore($sessionStates)
  useStore($activeSessionId)
  useStore($selectedStoredSessionId)
  useStore($connection)

  return resolveSessionContributionContext(input)
}
