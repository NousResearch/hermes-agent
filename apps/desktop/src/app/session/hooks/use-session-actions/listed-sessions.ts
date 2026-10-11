import { isMessagingSource, normalizeSessionSource } from '@/lib/session-source'
import { $projectTree } from '@/store/projects'
import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  sessionMatchesStoredId,
  setCronSessions,
  setMessagingSessions,
  setSessions,
  setUnlistedSessionOwnerRows
} from '@/store/session'
import {
  $removedSessionIds,
  type SessionTombstoneGenerationSnapshot,
  tombstoneLifecycleChanged
} from '@/store/session-removal'
import type { SessionInfo } from '@/types/hermes'

export type ListedSessionSlice = 'cron' | 'messaging' | 'sessions'

export function findListedSession(
  storedSessionId: string
): { session: SessionInfo; slice: ListedSessionSlice } | undefined {
  const match = (session: SessionInfo) => sessionMatchesStoredId(session, storedSessionId)
  const fromMessaging = $messagingSessions.get().find(match)

  if (fromMessaging) {
    return { session: fromMessaging, slice: 'messaging' }
  }

  const fromCron = $cronSessions.get().find(match)

  if (fromCron) {
    return { session: fromCron, slice: 'cron' }
  }

  const fromSessions = $sessions.get().find(match)

  if (fromSessions) {
    return { session: fromSessions, slice: 'sessions' }
  }

  return undefined
}

export function dropListedSession(storedSessionId: string): void {
  const keep = (session: SessionInfo) => !sessionMatchesStoredId(session, storedSessionId)

  setSessions(prev => prev.filter(keep))
  setMessagingSessions(prev => prev.filter(keep))
  setCronSessions(prev => prev.filter(keep))
  setUnlistedSessionOwnerRows(prev => prev.filter(keep))
}

export function listedSliceTarget(session: SessionInfo): ListedSessionSlice {
  return isMessagingSource(session.source)
    ? 'messaging'
    : normalizeSessionSource(session.source) === 'cron'
      ? 'cron'
      : 'sessions'
}

export function restoreListedSession(session: SessionInfo, slice?: ListedSessionSlice): void {
  const target: ListedSessionSlice = slice ?? listedSliceTarget(session)

  const prepend = (prev: SessionInfo[]) => [
    session,
    ...prev.filter(existing => !sessionMatchesStoredId(existing, session.id))
  ]

  if (target === 'messaging') {
    setMessagingSessions(prepend)

    return
  }

  if (target === 'cron') {
    setCronSessions(prepend)

    return
  }

  setSessions(prepend)
}

export function upsertResolvedSession(
  session: SessionInfo,
  storedSessionId: string,
  tombstoneGenerationsAtRequestStart: SessionTombstoneGenerationSnapshot
) {
  // Exact-id lookup intentionally resolves internal delegate children so a
  // watch tile can open them. They are not ordinary user conversations,
  // though, and the authoritative list endpoints omit them; caching one here
  // would bypass that boundary and leak it into the Sessions sidebar.
  if (session.is_internal_child) {
    return
  }

  const removed = $removedSessionIds.get()
  const identities = [storedSessionId, session.id, session._lineage_root_id]

  // A direct by-id resolve may have started just before an archive/delete
  // (#85163: the archive row click's bubbled resume raced the tombstone).
  // A stale response must not undo the optimistic eviction while the mutation's
  // tombstone is active, after the tombstone was already present at request
  // start, or after an add → remove ABA cycle made membership look unchanged.
  // Check every identity lineage-aware lookups use. This suppresses only the
  // sidebar-cache upsert: the resolved row is still returned so an explicit
  // resume-by-id can open archived history, and a later request after a
  // settled rollback can publish normally.
  if (
    session.archived ||
    identities.some(id => (id ? removed.has(id) : false)) ||
    tombstoneLifecycleChanged(tombstoneGenerationsAtRequestStart, identities)
  ) {
    return
  }

  const lineage = session._lineage_root_id ?? session.id

  // A hidden row (canonical Bot Chat, room plumbing) is unlisted by design:
  // inserting it into $sessions paints a sidebar row until the next refresh,
  // and the keep-list then holds it there (#113273). Park it on the off-list
  // owner atom the draft stubs ride — owner resolution still finds it via
  // ownerLookupSessionRows, the sidebar never does.
  if (session.hidden) {
    setUnlistedSessionOwnerRows(prev => [
      session,
      ...prev.filter(existing => (existing._lineage_root_id ?? existing.id) !== lineage)
    ])

    return
  }

  const prepend = (prev: SessionInfo[]) => [
    session,
    ...prev.filter(existing => {
      if (sessionMatchesStoredId(existing, storedSessionId)) {
        return false
      }

      return (existing._lineage_root_id ?? existing.id) !== lineage
    })
  ]

  // A resolve can observe a source move (cross-room /resume rewrites the row to
  // source='matrix', #113827): the row belongs to its current slice, and the
  // stale copy in every other slice must go or the session shows twice.
  // Identity-stable when nothing matched — every sidebar memo keys on these
  // arrays, and a resolve runs on each row open.
  const evict = (prev: SessionInfo[]) =>
    prev.some(existing => sessionMatchesStoredId(existing, storedSessionId))
      ? prev.filter(existing => !sessionMatchesStoredId(existing, storedSessionId))
      : prev

  const target = listedSliceTarget(session)

  setSessions(target === 'sessions' ? prepend : evict)
  setMessagingSessions(target === 'messaging' ? prepend : evict)
  setCronSessions(target === 'cron' ? prepend : evict)
}

// Every session row reachable through the profile-scoped project tree —
// preview rows on a collapsed project plus the drill-in lane rows. These are
// the only rows guaranteed to name their owning profile (the gateway stamps
// the request scope onto them), so owner resolution has to see them.
function projectTreeSessions(): SessionInfo[] {
  return $projectTree
    .get()
    .flatMap(project => [
      ...(project.previewSessions ?? []),
      ...project.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions))
    ])
}

// The best cached row for a stored id, across every list that can hold one.
// "Best" means self-describing: the same conversation can appear both as an
// ownerless legacy Recents copy and as a profile-stamped project-tree row, and
// picking the ownerless one throws away the only routing information we have.
export function cachedSessionRow(storedSessionId: string): SessionInfo | undefined {
  const candidates = [
    ...$sessions.get(),
    ...$cronSessions.get(),
    ...$messagingSessions.get(),
    ...projectTreeSessions()
  ].filter(session => sessionMatchesStoredId(session, storedSessionId))

  return (
    candidates.find(session => session.connection_id?.trim()) ??
    candidates.find(session => session.profile?.trim()) ??
    candidates[0]
  )
}
