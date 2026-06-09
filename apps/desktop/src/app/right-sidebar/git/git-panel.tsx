import { useCallback, useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { HermesGitStatusEntry, HermesGitStatusResult } from '@/global'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { RightSidebarSectionHeader } from '../section-header'

import { useGitStatus } from './use-git-status'

interface GitPanelProps {
  active: boolean
  cwd: string
  onOpenChange: (entry: HermesGitStatusEntry, gitRoot: string) => void
}

interface ChangeGroup {
  entries: HermesGitStatusEntry[]
  id: 'conflicts' | 'staged' | 'changes' | 'untracked'
  label: string
}

function groupChanges(status: HermesGitStatusResult, labels: Record<ChangeGroup['id'], string>): ChangeGroup[] {
  const conflicts = status.entries.filter(entry => entry.conflicted)
  const staged = status.entries.filter(entry => !entry.conflicted && Boolean(entry.index))
  const changes = status.entries.filter(entry => !entry.conflicted && Boolean(entry.worktree))
  const untracked = status.entries.filter(entry => entry.untracked)

  const groups: ChangeGroup[] = [
    { entries: conflicts, id: 'conflicts', label: labels.conflicts },
    { entries: staged, id: 'staged', label: labels.staged },
    { entries: changes, id: 'changes', label: labels.changes },
    { entries: untracked, id: 'untracked', label: labels.untracked }
  ]

  return groups.filter(group => group.entries.length > 0)
}

function statusCode(entry: HermesGitStatusEntry, group: ChangeGroup['id']) {
  if (group === 'conflicts') {return 'U'}

  if (group === 'untracked') {return 'U'}

  return group === 'staged' ? entry.indexCode : entry.worktreeCode
}

export function GitPanel({ active, cwd, onOpenChange }: GitPanelProps) {
  const { t } = useI18n()
  const { loading, refresh, status } = useGitStatus(cwd, active)

  const groups = useMemo(
    () =>
      groupChanges(status, {
        changes: t.rightSidebar.changes,
        conflicts: t.rightSidebar.mergeChanges,
        staged: t.rightSidebar.stagedChanges,
        untracked: t.rightSidebar.untrackedFiles
      }),
    [status, t]
  )

  const gitRoot = status.root || cwd

  const handleStage = useCallback(
    async (entry: HermesGitStatusEntry) => {
      if (!window.hermesDesktop?.gitStage) return
      await window.hermesDesktop.gitStage(gitRoot, entry.path)
      void refresh()
    },
    [gitRoot, refresh]
  )

  const handleUnstage = useCallback(
    async (entry: HermesGitStatusEntry) => {
      if (!window.hermesDesktop?.gitUnstage) return
      await window.hermesDesktop.gitUnstage(gitRoot, entry.path)
      void refresh()
    },
    [gitRoot, refresh]
  )

  const handleDiscard = useCallback(
    async (entry: HermesGitStatusEntry) => {
      if (!window.hermesDesktop?.gitDiscard) return
      await window.hermesDesktop.gitDiscard(gitRoot, entry.path)
      void refresh()
    },
    [gitRoot, refresh]
  )

  const branch = status.branch.name || (status.branch.detached ? status.branch.oid.slice(0, 8) : '')

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <RightSidebarSectionHeader>
        <span className="min-w-0 flex-1 truncate text-[0.6875rem] font-semibold uppercase tracking-[0.06em] text-(--ui-text-secondary)">
          {branch || t.rightSidebar.sourceControl}
        </span>
        {(status.branch.ahead > 0 || status.branch.behind > 0) && (
          <span className="mr-1 text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">
            {status.branch.ahead > 0 && `↑${status.branch.ahead}`}
            {status.branch.ahead > 0 && status.branch.behind > 0 && ' '}
            {status.branch.behind > 0 && `↓${status.branch.behind}`}
          </span>
        )}
        <span className="mr-1 text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{status.entries.length}</span>
        <Button
          aria-label={t.rightSidebar.refreshChanges}
          className="text-sidebar-foreground/70 hover:bg-sidebar-accent! hover:text-sidebar-accent-foreground!"
          disabled={!cwd || loading}
          onClick={() => void refresh()}
          size="icon-xs"
          variant="ghost"
        >
          <Codicon name="refresh" size="0.8125rem" spinning={loading} />
        </Button>
      </RightSidebarSectionHeader>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {!cwd ? (
          <GitEmpty body={t.rightSidebar.noProjectBody} title={t.rightSidebar.noProjectTitle} />
        ) : status.error ? (
          <GitEmpty body={status.error} title={t.rightSidebar.sourceControlUnavailable} />
        ) : !status.root ? (
          <GitEmpty body={t.rightSidebar.notGitRepositoryBody} title={t.rightSidebar.notGitRepository} />
        ) : groups.length === 0 ? (
          <GitEmpty body={t.rightSidebar.noChangesBody} title={t.rightSidebar.noChanges} />
        ) : (
          groups.map(group => (
            <section key={group.id}>
              <div className="sticky top-0 z-10 flex h-6 items-center bg-(--ui-sidebar-surface-background) px-2.5 text-[0.625rem] font-semibold uppercase tracking-[0.06em] text-(--ui-text-tertiary)">
                <span className="min-w-0 flex-1 truncate">{group.label}</span>
                <span className="tabular-nums text-(--ui-text-quaternary)">{group.entries.length}</span>
              </div>
              {group.entries.map(entry => {
                const pathParts = entry.path.split('/')
                const name = pathParts.pop() || entry.path
                const parent = pathParts.join('/')
                const code = statusCode(entry, group.id)
                const isStaged = group.id === 'staged'
                const isChange = group.id === 'changes' || group.id === 'untracked'

                return (
                  <div
                    className="group/change flex w-full min-w-0 items-center gap-0 hover:bg-(--ui-row-hover-background)"
                    key={`${group.id}:${entry.path}`}
                  >
                    {isStaged ? (
                      <button
                        aria-label={t.rightSidebar.unstageChanges}
                        className="flex shrink-0 items-center justify-center px-0.5 text-(--ui-text-quaternary) opacity-0 transition-opacity group-hover/change:opacity-100 hover:text-(--ui-text-primary)"
                        onClick={e => {
                          e.stopPropagation()
                          void handleUnstage(entry)
                        }}
                        title={t.rightSidebar.unstageChanges}
                        type="button"
                      >
                        <Codicon name="remove" size="0.75rem" />
                      </button>
                    ) : (
                      <>
                        <button
                          aria-label={t.rightSidebar.stageChanges}
                          className="flex shrink-0 items-center justify-center px-0.5 text-(--ui-text-quaternary) opacity-0 transition-opacity group-hover/change:opacity-100 hover:text-emerald-500"
                          onClick={e => {
                            e.stopPropagation()
                            void handleStage(entry)
                          }}
                          title={t.rightSidebar.stageChanges}
                          type="button"
                        >
                          <Codicon name="add" size="0.75rem" />
                        </button>
                        <button
                          aria-label={t.rightSidebar.discardChanges}
                          className="flex shrink-0 items-center justify-center px-0.5 text-(--ui-text-quaternary) opacity-0 transition-opacity group-hover/change:opacity-100 hover:text-rose-500"
                          onClick={e => {
                            e.stopPropagation()
                            void handleDiscard(entry)
                          }}
                          title={t.rightSidebar.discardChanges}
                          type="button"
                        >
                          <Codicon name="remove" size="0.75rem" />
                        </button>
                      </>
                    )}
                    <button
                      className="flex min-w-0 flex-1 items-center gap-2 py-1 text-left text-[0.6875rem]"
                      onClick={() => onOpenChange(entry, status.root || cwd)}
                      title={entry.originalPath ? `${entry.originalPath} → ${entry.path}` : entry.path}
                      type="button"
                    >
                      <Codicon className="shrink-0 text-(--ui-text-quaternary)" name="file" size="0.75rem" />
                      <span className="min-w-0 flex-1 truncate text-(--ui-text-secondary)">
                        {name}
                        {parent && <span className="ml-1 text-(--ui-text-quaternary)">{parent}</span>}
                      </span>
                      <span
                        className={cn(
                          'mr-1 w-4 shrink-0 text-center font-mono font-bold',
                          isStaged && 'text-emerald-500',
                          group.id === 'changes' && 'text-amber-500',
                          group.id === 'untracked' && 'text-emerald-500',
                          group.id === 'conflicts' && 'text-rose-500'
                        )}
                      >
                        {code}
                      </span>
                    </button>
                  </div>
                )
              })}
            </section>
          ))
        )}
      </div>
    </div>
  )
}

function GitEmpty({ body, title }: { body: string; title: string }) {
  return (
    <div className="flex h-full min-h-28 flex-col items-center justify-center gap-1 px-4 text-center">
      <div className="text-[0.7rem] font-semibold uppercase tracking-[0.07em] text-muted-foreground/75">{title}</div>
      <div className="text-[0.68rem] leading-relaxed text-muted-foreground/65">{body}</div>
    </div>
  )
}
