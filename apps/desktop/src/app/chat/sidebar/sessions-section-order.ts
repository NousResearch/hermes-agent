import type { SessionInfo } from '@/hermes'
import { flattenSessionsWithBranches, type SidebarSessionEntry } from '@/lib/session-branch-tree'

export function sessionEntriesForSection(sessions: readonly SessionInfo[], pinned: boolean): SidebarSessionEntry[] {
  // Pins are already in the user's persisted order. The branch flattener sorts
  // top-level groups by activity, which would undo a successful drag reorder.
  return pinned ? sessions.map(session => ({ session })) : flattenSessionsWithBranches(sessions)
}
