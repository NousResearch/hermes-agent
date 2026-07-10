// Public surface of the project/worktree sidebar, consumed by the sidebar root.
export { EnteredProjectContent } from './entered-content'
export { PROJECT_PREVIEW_COUNT, projectTreeCwd, sortProjectsForOverview, useRepoWorktreeMap } from './model'
export { ProjectBackRow, ProjectOverviewRow } from './overview-row'
export { ProjectMenu } from './project-menu'
export { ProjectSessionSortMenu } from './project-session-sort-menu'
export { flattenProjectSessions, type ProjectSessionSort, sortProjectSessions } from './session-sort'
export { SidebarWorkspaceGroup } from './workspace-group'
export {
  overlayLiveLanes,
  overlayLivePreviews,
  sessionRecency,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  type SidebarWorkspaceTree
} from './workspace-groups'
export { StartWorkButton } from './workspace-header'
export { isProjectSessionSort } from '@/lib/project-session-sort'
