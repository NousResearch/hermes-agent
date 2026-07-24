import type * as React from 'react'
import { useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { copyPath, revealPath } from '@/store/projects'

import { SidebarCount, SidebarRowLead } from '../chrome'

import { WorktreeDialog } from './worktree-dialog'

// Branch/worktree labels routinely share a long prefix (`bb/coding-context-…`),
// so plain end-truncation (`truncate`) hides exactly the suffix that tells two
// lanes apart — both render as "bb/coding-context…". Keep the tail pinned and
// ellipsize the HEAD instead, so `…context-facts-rpc` and `…context-persona`
// stay distinguishable. Falls back to whole-string for short labels.
function LaneLabel({ label, title }: { label: string; title?: string }) {
  const tailLen = Math.min(14, Math.floor(label.length / 2))
  const head = label.slice(0, label.length - tailLen)
  const tail = label.slice(label.length - tailLen)

  return (
    <span className="flex min-w-0" title={title}>
      <span className="truncate">{head}</span>
      <span className="shrink-0 whitespace-pre">{tail}</span>
    </span>
  )
}

// "+" affordance shared by repo and worktree headers — reveals on header hover.
export function WorkspaceAddButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <Tip label={label}>
      <button
        aria-label={label}
        className="grid size-4 shrink-0 place-items-center rounded-sm bg-transparent text-(--ui-text-quaternary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/workspace:opacity-100"
        onClick={onClick}
        type="button"
      >
        <Codicon name="add" size="0.75rem" />
      </button>
    </Tip>
  )
}

// Reveals the next page of already-loaded rows within a workspace/worktree.
export function WorkspaceShowMoreButton({
  count,
  label,
  onClick
}: {
  count: number
  label: string
  onClick: () => void
}) {
  const { t } = useI18n()
  const text = t.sidebar.showMoreIn(count, label)

  return (
    <Tip label={text}>
      <button
        aria-label={text}
        className="ml-auto grid size-5 place-items-center rounded-sm bg-transparent text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground"
        onClick={onClick}
        type="button"
      >
        <Codicon name="ellipsis" size="0.75rem" />
      </button>
    </Tip>
  )
}

// Per-worktree actions (linked worktree lanes only), mirroring the session row
// and ProjectMenu kebab: reveal in the file manager, copy path, and remove the
// worktree (runs a real `git worktree remove` via the caller's confirm dialog).
export function WorkspaceMenu({ path, onRemove }: { path: null | string; onRemove: () => void }) {
  const { t } = useI18n()
  const p = t.sidebar.projects

  return (
    <DropdownMenu>
      <Tip label={p.menu}>
        <DropdownMenuTrigger asChild>
          <button
            aria-label={p.menu}
            className="grid size-4 shrink-0 place-items-center rounded-sm bg-transparent text-(--ui-text-quaternary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/workspace:opacity-100 data-[state=open]:opacity-100"
            onClick={event => event.stopPropagation()}
            type="button"
          >
            <Codicon name="kebab-vertical" size="0.75rem" />
          </button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end" className="w-48" sideOffset={6}>
        <DropdownMenuItem disabled={!path} onSelect={() => void revealPath(path)}>
          <Codicon name="folder-opened" size="0.875rem" />
          <span>{p.reveal}</span>
        </DropdownMenuItem>
        <DropdownMenuItem disabled={!path} onSelect={() => void copyPath(path)}>
          <Codicon name="copy" size="0.875rem" />
          <span>{p.copyPath}</span>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={onRemove} variant="destructive">
          <Codicon name="trash" size="0.875rem" />
          <span>{`${p.removeWorktree}…`}</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// "New worktree": prompt for a branch name, then git spins up a fresh worktree
// for that branch under the repo (the lightest way) and we open a new session
// inside it. Naming is explicit — no auto-generated `hermes/work-<ts>` trees.
// The base branch defaults to the remote default (origin/HEAD); the user can
// pick any local or remote-tracking branch via a filterable combobox.
export function StartWorkButton({ repoPath, onStarted }: { repoPath: string; onStarted: (path: string) => void }) {
  const { t } = useI18n()
  const p = t.sidebar.projects
  const [open, setOpen] = useState(false)

  return (
    <>
      <Tip label={p.startWork}>
        <button
          aria-label={p.startWork}
          className="grid size-4 shrink-0 place-items-center rounded-sm bg-transparent text-(--ui-text-quaternary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/section:opacity-100 focus-visible:opacity-100"
          onClick={() => setOpen(true)}
          type="button"
        >
          <Codicon name="git-branch" size="0.75rem" />
        </button>
      </Tip>
      <WorktreeDialog onOpenChange={setOpen} onStarted={onStarted} open={open} repoPath={repoPath} />
    </>
  )
}

// Collapsible header shared by the repo (emphasis) and worktree levels: a toggle
// button with a leading glyph, plus an optional trailing action (the +).
export function WorkspaceHeader({
  action,
  count,
  emphasis = false,
  icon,
  label,
  onToggle,
  open,
  title
}: {
  action?: React.ReactNode
  count: React.ReactNode
  emphasis?: boolean
  icon: React.ReactNode
  label: string
  onToggle: () => void
  open: boolean
  /** Hover tooltip — the lane's full on-disk path (worktree / repo root). */
  title?: string
}) {
  return (
    <div
      className={cn(
        'group/workspace flex min-h-6 items-center gap-1 px-2 pt-1 text-[0.6875rem]',
        emphasis ? 'font-semibold text-(--ui-text-secondary)' : 'font-medium text-(--ui-text-tertiary)'
      )}
    >
      <button
        className={cn(
          'flex min-w-0 flex-1 items-center gap-1.5 bg-transparent text-left',
          emphasis ? 'hover:text-foreground' : 'hover:text-(--ui-text-secondary)'
        )}
        onClick={onToggle}
        type="button"
      >
        <SidebarRowLead>{icon}</SidebarRowLead>
        <LaneLabel label={label} title={title ? `${label}\n${title}` : label} />
        <span className="shrink-0">
          <SidebarCount>{count}</SidebarCount>
        </span>
        <DisclosureCaret
          className="shrink-0 text-(--ui-text-tertiary) opacity-0 transition group-hover/workspace:opacity-100"
          open={open}
        />
      </button>
      {action}
    </div>
  )
}
