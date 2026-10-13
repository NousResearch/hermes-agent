import { useStore } from '@nanostores/react'

import { $sidebarProjectRecentScopes, setSidebarProjectOrderIds, setSidebarProjectSortMode } from '@/store/layout'

import type { SidebarProjectTree } from './workspace-groups'

export type ProjectSortMode = 'manual' | 'recent'

// A project's sort choice belongs to the source AND profile shown in this window.
// No connection while switching: fall back to the existing manual behavior.
export const projectSortScopeKey = (connectionId: string, profileScope: string): string =>
  JSON.stringify([connectionId, profileScope])

export function projectSortModeForScope(
  recentScopes: Readonly<Record<string, string>>,
  connectionId: null | string,
  profileScope: string
): ProjectSortMode {
  return connectionId && recentScopes[projectSortScopeKey(connectionId, profileScope)] === 'recent'
    ? 'recent'
    : 'manual'
}

// The window's project order mode, plus the drag handler: persisting a manual
// drag order (orderByIds layers it over the default sort, so stale/new ids
// reconcile on the next render) also leaves Recent for this scope.
export function useProjectSortMode(connectionId: null | string, profileScope: string) {
  const mode = projectSortModeForScope(useStore($sidebarProjectRecentScopes), connectionId, profileScope)

  const reorderProjects = (ids: string[]) => {
    setSidebarProjectOrderIds(ids)

    if (mode === 'recent' && connectionId) {
      setSidebarProjectSortMode(projectSortScopeKey(connectionId, profileScope), 'manual')
    }
  }

  return { mode, reorderProjects }
}

// Strict user/assistant-message recency, not project or session activity.
// Missing qualified messages rank as zero; NEVER fall back to lastActive,
// which can include a heartbeat or tool message. Live overlays do not affect it.
export function sortProjectsByRecentActivity(projects: SidebarProjectTree[]): SidebarProjectTree[] {
  return [...projects].sort((a, b) => {
    if (Boolean(a.isNoProject) !== Boolean(b.isNoProject)) {
      return a.isNoProject ? -1 : 1
    }

    const aClock = a.lastMessageAt ?? 0
    const bClock = b.lastMessageAt ?? 0

    // A historical project may have no sessions in the bounded tree payload,
    // but its uncapped header clock still takes precedence over that count.
    if (aClock !== bClock) {
      return bClock - aClock
    }

    const aHasSessions = a.sessionCount > 0
    const bHasSessions = b.sessionCount > 0

    if (aHasSessions !== bHasSessions) {
      return aHasSessions ? -1 : 1
    }

    return a.label.localeCompare(b.label, undefined, { sensitivity: 'base' }) || a.id.localeCompare(b.id)
  })
}
