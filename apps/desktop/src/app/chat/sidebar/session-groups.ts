import type { SessionInfo } from '@/types/hermes'

export interface SidebarSessionGroup {
  id: string
  label: string
  path: null | string
  sessions: SessionInfo[]
  // Profile color for the ALL-profiles view; absent for workspace/source groups.
  color?: null | string
  loadingMore?: boolean
  mode?: 'profile' | 'source' | 'workspace'
  onLoadMore?: () => void
  totalCount?: number
  allowNewSession?: boolean
}

const WEBUI_GROUP_ID = 'source:webui'

const baseName = (path: string) =>
  path
    .replace(/[/\\]+$/, '')
    .split(/[/\\]/)
    .filter(Boolean)
    .pop()

function isWebUISession(session: SessionInfo): boolean {
  return String(session.source || '').trim().toLowerCase() === 'webui'
}

export function workspaceGroupsFor(sessions: SessionInfo[], noWorkspaceLabel = 'No workspace'): SidebarSessionGroup[] {
  const sourceGroups = new Map<string, SidebarSessionGroup>()
  const workspaceGroups = new Map<string, SidebarSessionGroup>()

  for (const session of sessions) {
    if (isWebUISession(session)) {
      const group = sourceGroups.get(WEBUI_GROUP_ID) ?? {
        allowNewSession: false,
        id: WEBUI_GROUP_ID,
        label: 'WebUI',
        mode: 'source' as const,
        path: null,
        sessions: []
      }

      group.sessions.push(session)
      sourceGroups.set(WEBUI_GROUP_ID, group)

      continue
    }

    const path = session.cwd?.trim() || ''
    const id = path || '__no_workspace__'
    const label = baseName(path) || path || noWorkspaceLabel

    const group = workspaceGroups.get(id) ?? { id, label, mode: 'workspace' as const, path: path || null, sessions: [] }
    group.sessions.push(session)
    workspaceGroups.set(id, group)
  }

  const groups = [...sourceGroups.values(), ...workspaceGroups.values()]

  // Groups keep recency order (Map insertion = first-seen in the recency-sorted
  // input, so an active project floats up). WebUI is a source group, not a
  // filesystem workspace, so it is promoted ahead of cwd groups to make imported
  // legacy conversations discoverable. Rows *within* each group sort by
  // creation time so they don't reshuffle every time a message lands.
  for (const group of groups) {
    group.sessions.sort((a, b) => b.started_at - a.started_at)
  }

  return groups
}
