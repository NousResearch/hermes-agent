import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useMemo, useState } from 'react'

import { type NewSessionSplitHandler, startNewSessionDrag } from '@/app/chat/new-session-drag'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { displayPath } from '@/lib/display-path'
import { $dismissedWorktreeIds, $removedWorktreeIds, dismissWorktree, setWorkspaceNodeOpen } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { removeWorktreePath, switchBranchInRepo } from '@/store/projects'

import { SidebarRowStack } from '../chrome'

import { PROJECT_SESSION_PAGE, SIDEBAR_GROUP_PAGE, useRevealedRows, useWorkspaceNodeOpen } from './model'
import { SidebarWorkspaceGroup } from './workspace-group'
import {
  mergeRepoWorktreeGroups,
  overlayRepoLanes,
  sessionRecency,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  type SidebarWorkspaceTree
} from './workspace-groups'
import { WorkspaceAddButton, WorkspaceHeader, WorkspaceShowMoreRow } from './workspace-header'

const mainCheckoutSessions = (groups: SidebarSessionGroup[]): SessionInfo[] => {
  const byId = new Map<string, SessionInfo>()

  for (const group of groups) {
    if (!group.isMain) {
      continue
    }

    for (const session of group.sessions) {
      const existing = byId.get(session.id)

      if (!existing || sessionRecency(session) > sessionRecency(existing)) {
        byId.set(session.id, session)
      }
    }
  }

  return [...byId.values()].sort((a, b) => sessionRecency(b) - sessionRecency(a) || a.id.localeCompare(b.id))
}

// The main-checkout lanes flatten into the project body (no redundant branch
// header), but keep the same bounded five-row lane page the grouped lanes had —
// a huge main-checkout history must not render all at once.
function EnteredMainSessionRows({
  groups,
  renderRows
}: {
  groups: SidebarSessionGroup[]
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
}) {
  const { t } = useI18n()
  const sessions = mainCheckoutSessions(groups)
  const lane = useRevealedRows(sessions, SIDEBAR_GROUP_PAGE)
  const mainGroup = groups.find(group => group.isMain && group.isHome) ?? groups.find(group => group.isMain)

  if (!lane.shown.length || !mainGroup) {
    return null
  }

  return (
    <>
      {renderRows(lane.shown)}
      {lane.more > 0 && (
        <WorkspaceShowMoreRow label={t.sidebar.showMoreIn(lane.more, mainGroup.label)} onClick={lane.showMore} />
      )}
    </>
  )
}

interface EnteredMainSessionButtonProps {
  project: SidebarProjectTree
  onNewSession: (path: null | string) => void
  repoWorktrees?: Record<string, HermesGitWorktree[]>
}

// The flattened main lane loses its per-lane WorkspaceAddButton, so the
// entered-project header carries branch-aware session creation instead: switch
// the repo to the visible main lane's branch, then open the new session there.
export function EnteredMainSessionButton({ project, onNewSession, repoWorktrees }: EnteredMainSessionButtonProps) {
  const { t } = useI18n()
  const repo = project.path ? project.repos.find(candidate => candidate.path === project.path) : undefined

  const mainGroup = useMemo(() => {
    if (!repo) {
      return undefined
    }

    const groups = mergeRepoWorktreeGroups(repo, repo.path ? repoWorktrees?.[repo.path] : undefined)

    return groups.find(group => group.isMain && group.isHome) ?? groups.find(group => group.isMain)
  }, [repo, repoWorktrees])

  if (!mainGroup?.path) {
    return null
  }

  const path = mainGroup.path

  const handleNewSession = async () => {
    try {
      await switchBranchInRepo(path, mainGroup.label)
    } catch (err) {
      notifyError(err, t.statusStack.coding.switchFailed(mainGroup.label))

      return
    }

    onNewSession(path)
  }

  return <WorkspaceAddButton label={t.sidebar.newSessionIn(mainGroup.label)} onClick={() => void handleNewSession()} />
}

// The entered project's body. Main-checkout sessions render directly — no
// redundant repo/branch header (the breadcrumb already names the project). Only
// linked worktrees nest, shown by branch. Multi-folder projects keep per-repo
// headers so the folders stay distinguishable.
export function EnteredProjectContent({
  project,
  renderRows,
  onNewSession,
  onNewSessionSplit,
  repoWorktrees,
  liveSessions,
  removedSessionIds
}: {
  project: SidebarProjectTree
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
  onNewSessionSplit?: NewSessionSplitHandler
  repoWorktrees?: Record<string, HermesGitWorktree[]>
  liveSessions?: SessionInfo[]
  removedSessionIds?: ReadonlySet<string>
}) {
  if (!project.repos.length) {
    return null
  }

  // Home's rows aren't anchored to a folder, so there's no repo or worktree
  // structure to show — just the chats, paged so a huge Home stays cheap.
  if (project.isNoProject) {
    return <HomeSessions project={project} renderRows={renderRows} />
  }

  const single = project.repos.length === 1

  return (
    <>
      {project.repos.map(repo => (
        <RepoFlatSection
          discoveredWorktrees={repo.path ? repoWorktrees?.[repo.path] : undefined}
          key={repo.id}
          liveSessions={liveSessions}
          onNewSession={onNewSession}
          onNewSessionSplit={onNewSessionSplit}
          removedSessionIds={removedSessionIds}
          renderRows={renderRows}
          repo={repo}
          showHeader={!single}
        />
      ))}
    </>
  )
}

function HomeSessions({
  project,
  renderRows
}: {
  project: SidebarProjectTree
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
}) {
  const { t } = useI18n()
  const sessions = useMemo(() => project.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions)), [project])
  const home = useRevealedRows(sessions, PROJECT_SESSION_PAGE)

  return (
    <>
      {renderRows(home.shown)}
      {home.more > 0 && (
        <WorkspaceShowMoreRow label={t.sidebar.showMoreIn(home.more, project.label)} onClick={home.showMore} />
      )}
    </>
  )
}

function RepoFlatSection({
  repo,
  showHeader,
  renderRows,
  onNewSession,
  onNewSessionSplit,
  discoveredWorktrees,
  liveSessions,
  removedSessionIds
}: {
  repo: SidebarWorkspaceTree
  showHeader: boolean
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
  onNewSessionSplit?: NewSessionSplitHandler
  discoveredWorktrees?: HermesGitWorktree[]
  liveSessions?: SessionInfo[]
  removedSessionIds?: ReadonlySet<string>
}) {
  const { t } = useI18n()
  const s = t.sidebar
  const [open, toggleOpen] = useWorkspaceNodeOpen(repo.id)
  const dismissedWorktrees = useStore($dismissedWorktreeIds)
  const removedWorktrees = useStore($removedWorktreeIds)

  // The repo's session lanes already come fully built from the backend; this
  // only injects empty VISUAL lanes from a live `git worktree list`.
  const mergedGroups = useMemo(() => mergeRepoWorktreeGroups(repo, discoveredWorktrees), [repo, discoveredWorktrees])

  // Optimistic placement runs against the MERGED lane set (backend + visual
  // git-worktree lanes) so out-of-tree/sibling worktrees — which exist as visual
  // lanes before the snapshot carries their sessions — get the new row. The
  // overlay drops lanes it empties, so re-merge to restore still-real worktrees.
  const overlaidGroups = useMemo(() => {
    if (!(liveSessions?.length || removedSessionIds?.size)) {
      return mergedGroups
    }

    const { groups } = overlayRepoLanes({ ...repo, groups: mergedGroups }, liveSessions ?? [], removedSessionIds)

    return mergeRepoWorktreeGroups({ id: repo.id, path: repo.path, groups }, discoveredWorktrees)
  }, [repo, mergedGroups, discoveredWorktrees, liveSessions, removedSessionIds])

  const discoveredWorktreePaths = useMemo(
    () =>
      new Set(
        (discoveredWorktrees ?? [])
          .map(worktree => worktree.path?.trim())
          .filter((path): path is string => Boolean(path))
      ),
    [discoveredWorktrees]
  )

  // Main lanes are always visible; linked worktrees can be user-dismissed.
  // Discovery may resurrect a removed worktree, never an explicit sidebar hide.
  const ordered = overlaidGroups.filter(
    group =>
      group.isMain ||
      !dismissedWorktrees.includes(group.id) ||
      (removedWorktrees.includes(group.id) && group.path && discoveredWorktreePaths.has(group.path))
  )

  const nestedGroups = ordered.filter(group => !group.isMain)

  // Removal asks how: actually `git worktree remove` it, or just hide the lane
  // and leave the worktree on disk. A dirty worktree escalates to a force prompt
  // instead of erroring (those changes are usually throwaway).
  const [removeTarget, setRemoveTarget] = useState<null | SidebarSessionGroup>(null)
  const [forceTarget, setForceTarget] = useState<null | SidebarSessionGroup>(null)

  const removeViaGit = async (group: SidebarSessionGroup, force = false) => {
    if (!repo.path || !group.path) {
      return
    }

    try {
      await removeWorktreePath(repo.path, group.path, { force })
      dismissWorktree(group.id, { removed: true })
    } catch (err) {
      // git refuses a non-force remove on a dirty/locked worktree — offer force
      // rather than dead-ending on an error toast.
      if (!force && /force|modified|untracked|dirty|locked|contains/i.test(String((err as Error)?.message ?? ''))) {
        setForceTarget(group)
      } else {
        notifyError(err, s.projects.removeWorktreeFailed)
      }
    }
  }

  const body = (
    <>
      <EnteredMainSessionRows groups={ordered} renderRows={renderRows} />
      {nestedGroups.map(group => (
        <SidebarWorkspaceGroup
          group={group}
          key={group.id}
          // The kanban bucket is read-only: it aggregates many task worktrees, so
          // "new session here" and "remove worktree" have no single target.
          onNewSession={group.isKanban ? undefined : onNewSession}
          onNewSessionSplit={group.isKanban ? undefined : onNewSessionSplit}
          onRemove={group.isMain || group.isKanban ? undefined : () => setRemoveTarget(group)}
          renderRows={renderRows}
        />
      ))}
    </>
  )

  // Both removal prompts share the shape (hide-from-sidebar + cancel + a
  // destructive action); only the copy and the destructive handler differ.
  const worktreeDialog = (
    target: null | SidebarSessionGroup,
    setTarget: (next: null | SidebarSessionGroup) => void,
    description: string,
    destructiveLabel: string,
    onDestructive: (group: SidebarSessionGroup) => void
  ) => (
    <ConfirmDialog
      confirmLabel={destructiveLabel}
      description={description}
      destructive
      // removeViaGit finishes on its own: it either dismisses the lane, escalates
      // to the force prompt, or toasts. Nothing left for the dialog to wait on.
      dismissOnConfirm
      onClose={() => setTarget(null)}
      onConfirm={() => {
        if (target) {
          onDestructive(target)
        }
      }}
      open={Boolean(target)}
      secondaryAction={{
        label: s.projects.removeFromSidebar,
        onClick: () => target && dismissWorktree(target.id)
      }}
      title={`${s.projects.removeWorktree} "${target?.label ?? ''}"?`}
    />
  )

  const removeDialog = (
    <>
      {worktreeDialog(
        removeTarget,
        setRemoveTarget,
        s.projects.removeWorktreeConfirm,
        s.projects.removeWorktree,
        group => void removeViaGit(group)
      )}
      {worktreeDialog(
        forceTarget,
        setForceTarget,
        s.projects.removeWorktreeDirty,
        s.projects.forceRemove,
        group => void removeViaGit(group, true)
      )}
    </>
  )

  if (!showHeader) {
    return (
      <>
        {body}
        {removeDialog}
      </>
    )
  }

  return (
    <SidebarRowStack>
      <WorkspaceHeader
        action={
          onNewSession && (
            <WorkspaceAddButton
              label={s.newSessionIn(repo.label)}
              onClick={() => {
                // Reveal the repo the new session targets if the user had it
                // collapsed — the session lands in one of its lanes.
                setWorkspaceNodeOpen(repo.id, true)
                onNewSession(repo.path)
              }}
              onPointerDown={
                onNewSessionSplit
                  ? event => {
                      startNewSessionDrag(
                        placement => {
                          setWorkspaceNodeOpen(repo.id, true)
                          onNewSessionSplit(placement.dir, {
                            anchor: placement.anchor,
                            before: placement.before,
                            cwd: repo.path
                          })
                        },
                        event,
                        { cwd: repo.path, label: s.newSessionIn(repo.label) }
                      )
                    }
                  : undefined
              }
            />
          )
        }
        emphasis
        icon={<Codicon className="shrink-0 text-(--ui-text-tertiary)" name="repo" size="0.75rem" />}
        label={repo.label}
        onToggle={toggleOpen}
        open={open}
        title={repo.path ? displayPath(repo.path) : undefined}
      />
      {open && <SidebarRowStack className="pl-2.5">{body}</SidebarRowStack>}
      {removeDialog}
    </SidebarRowStack>
  )
}
