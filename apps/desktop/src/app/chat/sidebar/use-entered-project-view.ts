import { useMemo } from 'react'

import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'
import { ALL_PROJECTS } from '@/store/project-scope'

import {
  excludeProjectSessions,
  overlayLiveLanes,
  reconcileEnteredProjectSessions,
  type SidebarProjectTree,
  type SidebarWorkspaceTree,
  useRepoWorktreeMap
} from './projects'
import { useEnteredProjectSessions } from './use-entered-project-sessions'

/** The sidebar's entered-project (drill-in) state, extracted from ChatSidebar. */
export interface EnteredProjectState {
  /** True when a concrete project scope is active (the drill-in view). */
  inProject: boolean
  /** The overview tree is active (projects present) and no project entered. */
  projectsActive: boolean
  enteredProjectId: string | undefined
  /** Hydrated + live-overlaid entered project; overview node while loading. */
  enteredProject: SidebarProjectTree | undefined
  /** The entered project with live `$sessions` overlaid onto its lanes. */
  enteredProjectContent: SidebarProjectTree | undefined
  /** Live sessions reconciled with the overview's preview rows. */
  enteredProjectOverlaySessions: SessionInfo[]
  projectLoadFailed: boolean
  projectLoading: boolean
  retryProject: () => void
  /** git worktree map for the entered project's repo paths (visual enhancer). */
  scopedRepoWorktrees: Record<string, HermesGitWorktree[]>
}

/**
 * Entered-project (drill-in) view state: lazy lane hydration, the overview
 * fallback while that fetch is in flight, and the live-session overlay that
 * keeps a just-created session visible before the backend snapshot folds it
 * in. Extracted from ChatSidebar into a topical sibling hook (code health:
 * the facade component only ratchets down).
 */
export function useEnteredProjectView(args: {
  agentProjectTree: SidebarProjectTree[] | undefined
  projectScope: string
  agentSessions: SessionInfo[]
  removedSessionIds: ReadonlySet<string>
  projectOwners: ReadonlyMap<string, string>
  orderRepos: (repos: SidebarWorkspaceTree[]) => SidebarWorkspaceTree[]
  isHiddenFromProjects: (session: SessionInfo) => boolean
  showAllProfiles: boolean
  gatewayReady: boolean
  projectTree: readonly SidebarProjectTree[]
  scopeKey: string
}): EnteredProjectState {
  const {
    agentProjectTree,
    projectScope,
    agentSessions,
    removedSessionIds,
    projectOwners,
    orderRepos,
    isHiddenFromProjects,
    showAllProfiles,
    gatewayReady,
    projectTree,
    scopeKey
  } = args

  const projectsActive = Boolean(agentProjectTree?.length)

  // The overview node for the entered project (structure + counts, empty lanes).
  const overviewEnteredProject =
    projectsActive && projectScope !== ALL_PROJECTS
      ? agentProjectTree?.find(node => node.id === projectScope)
      : undefined

  const inProject = Boolean(overviewEnteredProject)
  const enteredProjectId = overviewEnteredProject?.id

  // Entering a project lazily hydrates its full lanes (repo -> lane -> sessions)
  // from the backend — same grouping/ids as the overview, just with rows.
  const {
    project: enteredProjectTree,
    failed: projectLoadFailed,
    loading: projectLoading,
    retry: retryProject
  } = useEnteredProjectSessions(enteredProjectId, gatewayReady, projectTree, scopeKey)

  // Prefer the hydrated tree; fall back to the overview node (empty lanes) while
  // the drill-in fetch is in flight, so the header/structure render immediately.
  const enteredProject = useMemo<SidebarProjectTree | undefined>(() => {
    if (!overviewEnteredProject) {
      return undefined
    }

    const hydrated =
      enteredProjectTree && enteredProjectTree.id === overviewEnteredProject.id
        ? enteredProjectTree
        : overviewEnteredProject

    // The live-session overlay (creates/evictions) is applied per-repo in
    // RepoFlatSection, AFTER the visual git-worktree lanes are merged in (so
    // out-of-tree worktrees can be placed). Here we just order the snapshot and
    // drop pinned rows — the hydrated lanes come straight from the backend, so
    // they haven't been through projectModel's filter.
    // The label comes from the overview node either way — that's the model's
    // presentation copy (Home is translated there), not the raw payload's.
    return excludeProjectSessions(
      { ...hydrated, label: overviewEnteredProject.label, repos: orderRepos(hydrated.repos) },
      isHiddenFromProjects
    )
  }, [overviewEnteredProject, enteredProjectTree, orderRepos, isHiddenFromProjects])

  const enteredProjectOverlaySessions = useMemo(
    () => reconcileEnteredProjectSessions(agentSessions, overviewEnteredProject?.previewSessions),
    [agentSessions, overviewEnteredProject?.previewSessions]
  )

  // Overlay live `$sessions` onto the entered project so a just-created session
  // (which the backend snapshot hasn't folded in yet) counts as content and
  // renders immediately. Also carry over the overview's current preview rows:
  // its project tree and the separately hydrated drill-in can resolve at
  // different times, but a row visible in the overview must not disappear on
  // entry. The backend seeds each project folder as an (empty) repo, so the
  // overlay always has a lane to place a missing in-project session into.
  const enteredProjectContent = useMemo(
    () =>
      enteredProject
        ? overlayLiveLanes(enteredProject, enteredProjectOverlaySessions, removedSessionIds, projectOwners)
        : undefined,
    [enteredProject, enteredProjectOverlaySessions, removedSessionIds, projectOwners]
  )

  const scopedRepoPaths = useMemo(
    () =>
      enteredProject ? enteredProject.repos.map(repo => repo.path).filter((path): path is string => Boolean(path)) : [],
    [enteredProject]
  )

  // git worktree list is a VISUAL-only enhancer (empty lanes); never membership.
  const inEnteredProject = Boolean(enteredProject && !showAllProfiles)
  const [scopedRepoWorktrees] = useRepoWorktreeMap(scopedRepoPaths, inEnteredProject)

  return {
    inProject,
    projectsActive,
    enteredProjectId,
    enteredProject,
    enteredProjectContent,
    enteredProjectOverlaySessions,
    projectLoadFailed,
    projectLoading,
    retryProject,
    scopedRepoWorktrees
  }
}
