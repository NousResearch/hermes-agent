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
import { $dismissedWorktreeIds, dismissWorktree } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { removeWorktreePath } from '@/store/projects'

import { SidebarRowStack } from '../chrome'

import { useWorkspaceNodeOpen } from './model'
import { SidebarWorkspaceGroup } from './workspace-group'
import {
  mergeRepoWorktreeGroups,
  overlayRepoLanes,
  sessionRecency,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  type SidebarWorkspaceTree
} from './workspace-groups'
import { WorkspaceAddButton, WorkspaceHeader } from './workspace-header'

// Multi-folder projects first render one row per repo. A focused repo — or the
// only repo — then renders one complete, deduplicated session list across its
// primary and linked-worktree lanes.
export function EnteredProjectContent({
  project,
  renderRows,
  onNewSession,
  focusedRepoId,
  onFocusRepo,
  onExitRepo,
  repoWorktrees,
  liveSessions,
  removedSessionIds
}: {
  project: SidebarProjectTree
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
  focusedRepoId?: null | string
  onFocusRepo?: (repoId: string) => void
  onExitRepo?: () => void
  repoWorktrees?: Record<string, HermesGitWorktree[]>
  liveSessions?: SessionInfo[]
  removedSessionIds?: ReadonlySet<string>
}) {
  const { t } = useI18n()

  if (!project.repos.length) {
    return null
  }

  const singleRepo = project.repos.length === 1 ? project.repos[0] : null
  const focusedRepo = singleRepo ?? project.repos.find(repo => repo.id === focusedRepoId)

  if (!focusedRepo) {
    return (
      <SidebarRowStack>
        {project.repos.map(repo => (
          <div className="group/workspace flex min-h-7 items-center gap-1 px-2 text-[0.6875rem]" key={repo.id}>
            <button
              aria-label={`Open ${repo.label}`}
              className="flex min-w-0 flex-1 items-center gap-1.5 bg-transparent text-left font-semibold text-(--ui-text-secondary) hover:text-foreground"
              onClick={() => onFocusRepo?.(repo.id)}
              type="button"
            >
              <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="folder" size="0.75rem" />
              <span className="min-w-0 flex-1 truncate" title={repo.path ? `${repo.label}\n${repo.path}` : repo.label}>
                {repo.label}
              </span>
              <span className="shrink-0 text-[0.6875rem] font-medium text-(--ui-text-quaternary)">
                {repo.sessionCount}
              </span>
              <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="chevron-right" size="0.75rem" />
            </button>
            {onNewSession && (
              <WorkspaceAddButton label={t.sidebar.newSessionIn(repo.label)} onClick={() => onNewSession(repo.path)} />
            )}
          </div>
        ))}
      </SidebarRowStack>
    )
  }

  return (
    <>
      {!singleRepo && (
        <button
          className="flex min-h-7 w-full items-center gap-1.5 bg-transparent px-2 text-left text-[0.6875rem] font-medium text-(--ui-text-secondary) opacity-70 hover:opacity-100"
          onClick={onExitRepo}
          type="button"
        >
          <Codicon name="chevron-left" size="0.75rem" />
          <span className="truncate">{project.label}</span>
        </button>
      )}
      <div className="flex min-h-7 items-center gap-1 px-2 text-[0.6875rem] font-semibold text-(--ui-text-secondary)">
        <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="folder-opened" size="0.75rem" />
        <span className="min-w-0 flex-1 truncate" title={focusedRepo.path ?? undefined}>
          {focusedRepo.label}
        </span>
        <span className="shrink-0 text-[0.6875rem] font-medium text-(--ui-text-quaternary)">
          {focusedRepo.sessionCount}
        </span>
        {onNewSession && (
          <WorkspaceAddButton
            label={t.sidebar.newSessionIn(focusedRepo.label)}
            onClick={() => onNewSession(focusedRepo.path)}
          />
        )}
      </div>
      <RepoFlatSection
        discoveredWorktrees={focusedRepo.path ? repoWorktrees?.[focusedRepo.path] : undefined}
        flattenSessions
        liveSessions={liveSessions}
        onNewSession={onNewSession}
        removedSessionIds={removedSessionIds}
        renderRows={renderRows}
        repo={focusedRepo}
        showHeader={false}
      />
    </>
  )
}

function RepoFlatSection({
  repo,
  showHeader,
  renderRows,
  onNewSession,
  discoveredWorktrees,
  flattenSessions = false,
  liveSessions,
  removedSessionIds
}: {
  repo: SidebarWorkspaceTree
  showHeader: boolean
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
  discoveredWorktrees?: HermesGitWorktree[]
  flattenSessions?: boolean
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

  const repoCount = ordered.reduce((sum, group) => sum + group.sessions.length, 0)

  const flattenedSessions = useMemo(() => {
    const byId = new Map<string, SessionInfo>()

    for (const group of ordered) {
      for (const session of group.sessions) {
        byId.set(session.id, byId.get(session.id) ?? session)
      }
    }

    return [...byId.values()].sort((a, b) => sessionRecency(b) - sessionRecency(a))
  }, [ordered])

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

  const body = flattenSessions ? (
    renderRows(flattenedSessions)
  ) : (
    <>
      {ordered.map(group => (
        <SidebarWorkspaceGroup
          group={group}
          key={group.id}
          // The kanban bucket is read-only: it aggregates many task worktrees, so
          // "new session here" and "remove worktree" have no single target.
          onNewSession={group.isKanban ? undefined : onNewSession}
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
            <WorkspaceAddButton label={s.newSessionIn(repo.label)} onClick={() => onNewSession(repo.path)} />
          )
        }
        count={repoCount}
        emphasis
        icon={<Codicon className="shrink-0 text-(--ui-text-tertiary)" name="repo" size="0.75rem" />}
        label={repo.label}
        onToggle={toggleOpen}
        open={open}
        title={repo.path ?? undefined}
      />
      {open && <SidebarRowStack className="pl-2.5">{body}</SidebarRowStack>}
      {removeDialog}
    </SidebarRowStack>
  )
}
