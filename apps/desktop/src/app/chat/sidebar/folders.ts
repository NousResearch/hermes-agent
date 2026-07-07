import type { SessionInfo } from '@/hermes'
import type { SidebarFolder } from '@/store/sidebar-folders'

// Folders visible in the given profile scope: a folder shows when it was
// explicitly pinned global, or when it belongs to the active profile.
export function visibleFoldersForScope(folders: SidebarFolder[], scope: string): SidebarFolder[] {
  return folders.filter(folder => folder.pinned || folder.profileKey === scope)
}

// Resolve a folder's stored session keys to live sessions, dropping any key that
// doesn't resolve — stale ids, sessions not yet loaded, or sessions outside the
// current profile scope (since `sessionByAnyId` is built from the visible set).
// This is what keeps a folder with non-loaded members from crashing the render.
// Order follows the folder's own `sessionIds`.
export function resolveFolderSessions(
  folder: SidebarFolder,
  sessionByAnyId: Map<string, SessionInfo>
): SessionInfo[] {
  const seen = new Set<string>()
  const out: SessionInfo[] = []

  for (const key of folder.sessionIds) {
    const session = sessionByAnyId.get(key)

    if (session && !seen.has(session.id)) {
      seen.add(session.id)
      out.push(session)
    }
  }

  return out
}

// Drop any session whose durable key is claimed by a folder, so foldered
// sessions disappear from the unfiled list (and, by extension, from the
// workspace tree and recents count that derive from it). Pinned is computed
// from a separate source and is intentionally not filtered here.
export function filterUnfiledSessions(
  sessions: SessionInfo[],
  folderedKeys: Set<string>,
  keyOf: (session: SessionInfo) => string
): SessionInfo[] {
  if (folderedKeys.size === 0) {
    return sessions
  }

  return sessions.filter(session => !folderedKeys.has(keyOf(session)))
}
