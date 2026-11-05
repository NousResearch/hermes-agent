import type { SessionInfo } from '@/types/hermes'

export interface ProfileSessionRecoveryScope {
  connectionId?: null | string
  profile: string
}

type RecoveryRow = Pick<
  SessionInfo,
  'connection_id' | 'id' | 'last_active' | 'message_count' | 'profile' | 'started_at'
>

/**
 * Pick the newest real conversation for one exact Desktop owner scope.
 *
 * Rows from an ambient (non-registry) list have no connection_id; they belong
 * to the connection that supplied the list. Aggregated rows carry both fields,
 * so a profile name alone is never enough to prove ownership.
 */
export function mostRecentSessionWithMessages<T extends RecoveryRow>(
  sessions: readonly T[],
  scope: ProfileSessionRecoveryScope
): T | undefined {
  const profile = scope.profile.trim() || 'default'
  const connectionId = scope.connectionId?.trim() || null

  return sessions
    .filter(session => {
      if (session.message_count <= 0 || (session.profile?.trim() || 'default') !== profile) {
        return false
      }

      // Only rows with explicit provenance can satisfy a requested registry
      // connection. Untagged rows are valid only for the ambient scope.
      return connectionId === null ? !session.connection_id : session.connection_id?.trim() === connectionId
    })
    .sort((a, b) => b.last_active - a.last_active || b.started_at - a.started_at || b.id.localeCompare(a.id))[0]
}

export function recoverMostRecentProfileSession<T extends RecoveryRow>(params: {
  hasNonEmptyDraft: boolean
  onResume: (session: T) => void
  scope: ProfileSessionRecoveryScope
  sessions: readonly T[]
}): boolean {
  if (params.hasNonEmptyDraft) {
    return false
  }

  const session = mostRecentSessionWithMessages(params.sessions, params.scope)

  if (!session) {
    return false
  }

  params.onResume(session)

  return true
}
