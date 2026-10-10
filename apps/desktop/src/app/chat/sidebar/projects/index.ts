// Public surface of the project/worktree sidebar, consumed by the sidebar root.
export { EnteredProjectContent } from './entered-content'
export {
  nestProjectsByParent,
  orderProjectsByIds,
  PROJECT_PREVIEW_COUNT,
  projectSubtreeSessionIds,
  projectTreeCwd,
  sortProjectsForOverview,
  useRepoWorktreeMap,
  visibleProjectRows
} from './model'
export { ProjectBackRow, ProjectOverviewRow } from './overview-row'
export { ProjectMenu } from './project-menu'
export { useProjectRowData } from './row-data'
export { SidebarWorkspaceGroup } from './workspace-group'
export {
  excludeProjectSessions,
  liveSessionProjectId,
  liveSessionsForProject,
  overlayLiveLanes,
  overlayLivePreviews,
  projectOwnerBySessionId,
  reconcileEnteredProjectSessions,
  sessionBucketId,
  sessionMatchesProjectFilter,
  sessionRecency,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  type SidebarWorkspaceTree
} from './workspace-groups'
export { StartWorkButton } from './workspace-header'
