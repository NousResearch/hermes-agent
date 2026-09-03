import { useCallback, useEffect, useRef, useState } from 'react'

import { getApiRequestConnection } from '@/api/client'
import { listAllProfileSessions, type SessionInfo } from '@/hermes'
import { requestSessionResume, setSessionOwnerHint, $sessions as sidebarSessions } from '@/store/session'
import type { SessionOwnerRoute } from '@/store/session-request-router'

import { openSession, type OpenSessionNavigate } from '../open-session'

const profileKey = (session: Pick<SessionInfo, 'profile'>): string => session.profile?.trim() || 'default'
const connectionKey = (session: Pick<SessionInfo, 'connection_id'>): string => session.connection_id?.trim() || 'local'

const ownerKey = (owner: { connection_id?: string; profile?: string }): string =>
  JSON.stringify([owner.connection_id?.trim() || 'local', owner.profile?.trim() || 'default'])

const durableId = (session: Pick<SessionInfo, '_lineage_root_id' | 'connection_id' | 'id' | 'profile'>): string =>
  JSON.stringify([connectionKey(session), profileKey(session), session._lineage_root_id || session.id])

const recency = (session: SessionInfo): number => Math.max(session.last_active || 0, session.started_at || 0)

export function commandCenterSessionOwnerRoute(session: SessionInfo): SessionOwnerRoute {
  return {
    connectionId: connectionKey(session),
    profile: profileKey(session)
  }
}

/**
 * Open an aggregate row through the owner captured on that exact row.
 *
 * Force the main surface rather than focusing a same-id tile: stored ids can be
 * duplicated by copied profile stores, while tile identity is historically
 * id-only. The explicit resume event carries the owner route even when the URL
 * is already `/session/:id`, so a same-id selection cannot reuse another
 * profile's runtime or rely on an ambient socket.
 */
export function openCommandCenterSession(session: SessionInfo, navigate: OpenSessionNavigate): void {
  const owner = commandCenterSessionOwnerRoute(session)

  setSessionOwnerHint(session.id, owner)
  openSession(session.id, navigate, 'main')
  requestSessionResume(session.id, owner)
}

/**
 * Deterministic command-center view over cross-profile session rows.
 *
 * A stored id is only durable inside its owning connection + profile. Copied
 * profile databases and separate registered gateways may legitimately carry the
 * same id; those rows remain distinct. Duplicate observations of the same exact
 * owner/lineage collapse to the freshest row.
 */
export function mergeCommandCenterSessions(
  previous: SessionInfo[],
  incoming: SessionInfo[],
  errors: Array<{ connection_id?: string; error?: string; profile?: string }> = []
): SessionInfo[] {
  const failedConnections = new Set(
    errors.filter(error => error.profile?.trim() === 'all').map(error => error.connection_id?.trim() || 'local')
  )

  const failedOwners = new Set(errors.filter(error => error.profile?.trim() !== 'all').map(ownerKey))
  const byIdentity = new Map<string, SessionInfo>()

  const retained = previous.filter(
    row => failedConnections.has(connectionKey(row)) || failedOwners.has(ownerKey(row))
  )

  for (const session of [...retained, ...incoming]) {
    const key = durableId(session)
    const current = byIdentity.get(key)

    if (!current || recency(session) >= recency(current)) {
      byIdentity.set(key, session)
    }
  }

  return [...byIdentity.entries()]
    .sort(([leftKey, left], [rightKey, right]) => recency(right) - recency(left) || leftKey.localeCompare(rightKey))
    .map(([, session]) => session)
}

export function useCommandCenterSessions(enabled: boolean) {
  // Paint the already-known foreground/default rows immediately. The aggregate
  // replaces this seed when it lands; a missing/older backend leaves the
  // existing Desktop history usable instead of flashing an empty pane.
  const [sessions, setSessions] = useState<SessionInfo[]>(() => sidebarSessions.get())
  const [errors, setErrors] = useState<Array<{ error: string; profile: string }>>([])
  const [loading, setLoading] = useState(false)
  const requestRef = useRef(0)

  const refresh = useCallback(async () => {
    const requestId = requestRef.current + 1
    requestRef.current = requestId
    setLoading(true)

    try {
      const result = await listAllProfileSessions(500, 0, 'include', 'recent', 'all')

      if (requestRef.current !== requestId) {
        return
      }

      const requestConnection = getApiRequestConnection() ?? 'local'

      const nextErrors = (result.errors ?? []).map(error => ({
        ...error,
        connection_id: error.connection_id || requestConnection
      }))

      setSessions(previous => mergeCommandCenterSessions(previous, result.sessions ?? [], nextErrors))
      setErrors(nextErrors)
    } catch (error) {
      if (requestRef.current === requestId) {
        setErrors([{ error: error instanceof Error ? error.message : String(error), profile: 'all' }])
      }
    } finally {
      if (requestRef.current === requestId) {
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    if (enabled) {
      void refresh()
    }
  }, [enabled, refresh])

  return { errors, loading, refresh, sessions }
}
