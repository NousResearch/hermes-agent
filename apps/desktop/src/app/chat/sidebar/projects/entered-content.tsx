import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { displayPath } from '@/lib/display-path'
import { $dismissedWorktreeIds, dismissWorktree, setWorkspaceNodeOpen } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { removeWorktreePath, switchBranchInRepo } from '@/store/projects'

import { SidebarRowStack } from '../chrome'

import { SIDEBAR_GROUP_PAGE, useWorkspaceNodeOpen } from './model'
import { SidebarWorkspaceGroup } from './workspace-group'
import {
  mergeRepoWorktreeGroups,
  overlayRepoLanes,
  sessionRecency,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  type SidebarWorkspaceTree
} from './workspace-groups'
import { WorkspaceAddButton, WorkspaceHeader, WorkspaceShowMoreButton } from './workspace-header'

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

function EnteredMainSessionRows({
  groups,
  renderRows
}: {
  groups: SidebarSessionGroup[]
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
}) {
  const [visibleCount, setVisibleCount] = useState(SIDEBAR_GROUP_PAGE)
  const sessions = mainCheckoutSessions(groups)
  const mainGroup = groups.find(group => group.isMain && group.isHome) ?? groups.find(group => group.isMain)

  if (!sessions.length || !mainGroup) {
    return null
  }

  const visibleSessions = sessions.slice(0, visibleCount)
  const hiddenCount = sessions.length - visibleSessions.length
  const nextCount = Math.min(SIDEBAR_GROUP_PAGE, hiddenCount)

  return (
    <>
      {renderRows(visibleSessions)}
      {hiddenCount > 0 && (
        <WorkspaceShowMoreButton
          count={nextCount}
          label={mainGroup.label}
          onClick={() => setVisibleCount(count => count + SIDEBAR_GROUP_PAGE)}
        />
      )}
    </>
  )
}

interface EnteredMainSessionButtonProps {
  project: SidebarProjectTree
  onNewSession: (path: null | string) => void
  repoWorktrees?: Record<string, HermesGitWorktree[]>
}

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
  repoWorktrees,
  liveSessions,
  removedSessionIds
}: {
  project: SidebarProjectTree
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
  repoWorktrees?: Record<string, HermesGitWorktree[]>
  liveSessions?: SessionInfo[]
  removedSessionIds?: ReadonlySet<string>
}) {
  if (!project.repos.length) {
    return null
  }

  // Home's rows aren't anchored to a folder, so there's no repo or worktree
  // structure to show — just the chats.
  if (project.isNoProject) {
    return <>{renderRows(project.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions)))}</>
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
          removedSessionIds={removedSessionIds}
          renderRows={renderRows}
          repo={repo}
          showHeader={!single}
        />
      ))}
    </>
  )
}

function RepoFlatSection({
  repo,
  showHeader,
  renderRows,
  onNewSession,
  discoveredWorktrees,
  liveSessions,
  removedSessionIds
}: {
  repo: SidebarWorkspaceTree
  showHeader: boolean
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
  discoveredWorktrees?: HermesGitWorktree[]
  liveSessions?: SessionInfo[]
  removedSessionIds?: ReadonlySet<string>
}) {
  const { t } = useI18n()
  const s = t.sidebar
  const [open, toggleOpen] = useWorkspaceNodeOpen(repo.id)
  const dismissedWorktrees = useStore($dismissedWorktreeIds)

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
  // A live `git worktree list` hit wins over an old dismissal: if git says the
  // worktree exists again (or still exists after "hide from sidebar"), surface it.
  const ordered = overlaidGroups.filter(
    group =>
      group.isMain || !dismissedWorktrees.includes(group.id) || (group.path && discoveredWorktreePaths.has(group.path))
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
      dismissWorktree(group.id)
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
          onRemove={group.isKanban ? undefined : () => setRemoveTarget(group)}
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
    <Dialog onOpenChange={isOpen => !isOpen && setTarget(null)} open={Boolean(target)}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{`${s.projects.removeWorktree} "${target?.label ?? ''}"?`}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button onClick={() => setTarget(null)} variant="ghost">
            {t.common.cancel}
          </Button>
          <Button
            onClick={() => {
              if (target) {
                dismissWorktree(target.id)
              }

              setTarget(null)
            }}
            variant="secondary"
          >
            {s.projects.removeFromSidebar}
          </Button>
          <Button
            onClick={() => {
              setTarget(null)

              if (target) {
                onDestructive(target)
              }
            }}
            variant="destructive"
          >
            {destructiveLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
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
