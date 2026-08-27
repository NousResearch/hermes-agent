/**
 * Deleted-connection roster hygiene: the pure half of the #93492 sweep.
 *
 * Marking a group-chat member descriptor whose connection is gone does not
 * depend on where the rooms live, so it sits below the room store — the sweep
 * in plugin.tsx drives these against the live atom.
 */

import type { GroupChat, GroupMember } from './types'

// ── deleted-connection roster hygiene (#93492 root cause) ───────────────────
// Deleting a cloud/remote connection used to leave every persisted group-chat
// member descriptor that referenced it behind untouched. Those orphaned rows
// (remoteSource: true, connection gone) are exactly the shape that made
// render-path route lookups throw "Bot X has no connection owner" on every
// group open, permanently — the poisoned row lives in plugin storage. The
// sweep below runs on the registry's 'removed' push (and its annotate helper
// again at hydrate for rows orphaned before this build). It never hard-deletes
// user data: the member row is kept and marked, so panes render the existing
// degraded 'Gateway removed' botSourceStatus state instead of crashing.

/** Keep the member's identity; mark it so botSourceStatus reads
 *  'Gateway removed' and no render-path route lookup can throw on it. */
export function markOrphanedGroupMemberDescriptor(member: GroupMember): GroupMember {
  return {
    ...member,
    sourceMissing: true,
    sourceReachable: false
  }
}
export function groupMemberReferencesConnection(member: GroupMember, connectionId: string) {
  const id = String(connectionId || '').trim()
  if (!id) {
    return false
  }
  return String(member?.connectionId || '').trim() === id || String(member?.route?.connectionId || '').trim() === id
}

/** Hydrate-time pass for rows orphaned BEFORE this build (the poisoned rows
 *  that made #93492 survive app restarts). Two shapes are annotated, never
 *  deleted:
 *  - a descriptor that already lost its connectionId (route unresolvable —
 *    exactly what a stale row looks like once its connection was deleted);
 *  - a descriptor whose connectionId is absent from the live registry, when
 *    the caller could obtain one (liveConnectionIds === null means "registry
 *    unavailable", which must NOT read as "everything is orphaned").
 *  Pure on the rooms map; returns { rooms, changed }. */
export function annotateOrphanedGroupChatMembers(
  rooms: Record<string, GroupChat>,
  liveConnectionIds: ReadonlySet<string> | null = null
) {
  // Duck-typed, not instanceof: callers (and vm-based tests) may hand a Set
  // constructed in another realm.
  const live = liveConnectionIds && typeof liveConnectionIds.has === 'function' ? liveConnectionIds : null
  const next: Record<string, GroupChat> = {}
  let changed = false
  for (const [name, room] of Object.entries(rooms || {})) {
    const members = Array.isArray(room?.members) ? room.members : []
    const orphaned = (member: GroupMember) => {
      if (!member || member.sourceMissing) {
        return false
      }
      if (!member.sourceScoped && !member.remoteSource) {
        return false
      }
      const id = String(member.route?.connectionId || member.connectionId || '').trim()
      if (!id) {
        // Route unresolvable: this is the row shape that threw on render.
        return true
      }
      return live ? !live.has(id) : false
    }
    if (!members.some(orphaned)) {
      next[name] = room
      continue
    }
    changed = true
    next[name] = {
      ...room,
      members: members.map(member => (orphaned(member) ? markOrphanedGroupMemberDescriptor(member) : member))
    }
  }
  return {
    rooms: next,
    changed
  }
}
