import { atom, computed } from 'nanostores'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'

// The project tree atom lives in this leaf so store/session-states.ts can
// read it (a focused tile's session is often older than the paginated recents
// page, so the tree is the only loaded copy of its row, #76535) without
// pulling store/projects.ts — and through it store/gateway.ts — into every
// suite that partially mocks the gateway. store/projects.ts owns all writes
// and re-exports the atom so existing call sites keep their import path.
export const $projectTree = atom<SidebarProjectTree[]>([])

/** Every session row the loaded tree carries, flattened for owner lookups.
 *  Some sources are listed ONLY through the project-grouped view (ACP:
 *  recents excludes editor-driven rows and the messaging slice drops every
 *  local source), so the tree is the only loaded copy of their row — without
 *  it the workspace tab can't resolve the session and falls through to the
 *  "New session" draft placeholder (#137048). Computed so the flatten runs
 *  once per tree change, not once per lookup. */
export const $projectTreeSessionRows = computed($projectTree, tree =>
  tree.flatMap(project => [
    ...(project.previewSessions ?? []),
    ...project.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions))
  ])
)
