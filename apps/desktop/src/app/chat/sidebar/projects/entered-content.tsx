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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { $dismissedWorktreeIds, dismissWorktree } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { removeWorktreePath, switchBranchInRepo } from '@/store/projects'

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
import { WorkspaceAddButton, WorkspaceHeader, WorkspaceMenuItems } from './workspace-header'

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
              aria-label={repo.label}
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
      <RepoFlatSection
        discoveredWorktrees={focusedRepo.path ? repoWorktrees?.[focusedRepo.path] : undefined}
        flattenSessions
        liveSessions={liveSessions}
        onNewSession={onNewSession}
        removedSessionIds={removedSessionIds}
        renderHeader={actions => (
          <div className="group/workspace flex min-h-7 items-center gap-1 px-2 text-[0.6875rem] font-semibold text-(--ui-text-secondary)">
            <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="folder-opened" size="0.75rem" />
            <span className="min-w-0 flex-1 truncate" title={focusedRepo.path ?? undefined}>
              {focusedRepo.label}
            </span>
            <span className="shrink-0 text-[0.6875rem] font-medium text-(--ui-text-quaternary)">
              {focusedRepo.sessionCount}
            </span>
            {actions}
          </div>
        )}
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
  renderHeader,
  liveSessions,
  removedSessionIds
}: {
  repo: SidebarWorkspaceTree
  showHeader: boolean
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
  discoveredWorktrees?: HermesGitWorktree[]
  flattenSessions?: boolean
  renderHeader?: (actions: React.ReactNode) => React.ReactNode
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

  const header = renderHeader?.(
    <RepoActionsMenu
      groups={ordered}
      onNewSession={onNewSession}
      onRemove={setRemoveTarget}
      repoLabel={repo.label}
      repoPath={repo.path}
    />
  )

  if (!showHeader) {
    return (
      <>
        {header}
        {body}
        {removeDialog}
      </>
    )
  }

  return (
    <SidebarRowStack>
      {header}
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

function RepoActionsMenu({
  groups,
  repoLabel,
  repoPath,
  onNewSession,
  onRemove
}: {
  groups: SidebarSessionGroup[]
  repoLabel: string
  repoPath: null | string
  onNewSession?: (path: null | string) => void
  onRemove: (group: SidebarSessionGroup) => void
}) {
  const { t } = useI18n()
  const targets = groups.filter(group => !group.isKanban)
  const worktrees = groups.filter(group => !group.isMain && !group.isKanban)

  const startSession = async (group?: SidebarSessionGroup) => {
    if (!onNewSession) {
      return
    }

    if (group?.isMain && group.path && group.label) {
      try {
        await switchBranchInRepo(group.path, group.label)
      } catch (err) {
        notifyError(err, t.statusStack.coding.switchFailed(group.label))

        return
      }
    }

    onNewSession(group?.path ?? repoPath)
  }

  if (targets.length <= 1 && worktrees.length === 0) {
    return onNewSession ? (
      <WorkspaceAddButton label={t.sidebar.newSessionIn(repoLabel)} onClick={() => void startSession(targets[0])} />
    ) : null
  }

  if (!onNewSession && worktrees.length === 0) {
    return null
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          aria-label={onNewSession ? t.sidebar.newSessionIn(repoLabel) : t.sidebar.projects.menu}
          className="grid size-4 shrink-0 place-items-center rounded-sm bg-transparent text-(--ui-text-quaternary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/workspace:opacity-100 data-[state=open]:opacity-100"
          type="button"
        >
          <Codicon name={onNewSession ? 'add' : 'kebab-vertical'} size="0.75rem" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52" sideOffset={6}>
        {onNewSession &&
          targets.map(group => (
            <DropdownMenuItem key={`new:${group.id}`} onSelect={() => void startSession(group)}>
              <Codicon name={group.isHome ? 'home' : 'git-branch'} size="0.875rem" />
              <span>{t.sidebar.newSessionIn(group.label)}</span>
            </DropdownMenuItem>
          ))}
        {onNewSession && worktrees.length > 0 && <DropdownMenuSeparator />}
        {worktrees.map(group => (
          <DropdownMenuSub key={`actions:${group.id}`}>
            <DropdownMenuSubTrigger>
              <Codicon name="git-branch" size="0.875rem" />
              <span>{group.label}</span>
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <WorkspaceMenuItems onRemove={() => onRemove(group)} path={group.path} />
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
