import type * as React from 'react'
import { useCallback, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { SanitizedInput } from '@/components/ui/sanitized-input'
import type { HermesGitBranch } from '@/global'
import { useI18n } from '@/i18n'
import { gitRef } from '@/lib/sanitize'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { copyPath, listRepoBranches, revealPath, startWorkInRepo } from '@/store/projects'

import { SidebarCount, SidebarRowLead } from '../chrome'

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
    <button
      aria-label={label}
      className="grid size-4 shrink-0 place-items-center rounded-sm bg-transparent text-(--ui-text-quaternary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/workspace:opacity-100"
      onClick={onClick}
      type="button"
    >
      <Codicon name="add" size="0.75rem" />
    </button>
  )
}

// Reveals the next page of already-loaded rows within a workspace/worktree.
export function WorkspaceShowMoreButton({ count, label, onClick }: { count: number; label: string; onClick: () => void }) {
  const { t } = useI18n()
  const text = t.sidebar.showMoreIn(count, label)

  return (
    <button
      aria-label={text}
      className="ml-auto grid size-5 place-items-center rounded-sm bg-transparent text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground"
      onClick={onClick}
      type="button"
    >
      <Codicon name="ellipsis" size="0.75rem" />
    </button>
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
export function StartWorkButton({ repoPath, onStarted }: { repoPath: string; onStarted: (path: string) => void }) {
  const { t } = useI18n()
  const s = t.sidebar
  const p = s.projects
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [pending, setPending] = useState(false)
  // "Convert an existing branch into a worktree" sub-mode: the body swaps the
  // new-branch name input for a filterable list of the repo's branches.
  const [convertMode, setConvertMode] = useState(false)
  const [branches, setBranches] = useState<HermesGitBranch[]>([])
  const [branchesLoading, setBranchesLoading] = useState(false)

  // Pull the repo's branches each time the picker is entered (cheap + bounded),
  // so a branch created mid-session shows up.
  const loadBranches = useCallback(async () => {
    if (!repoPath) {
      return
    }

    setBranchesLoading(true)

    try {
      setBranches(await listRepoBranches(repoPath))
    } catch {
      setBranches([])
    } finally {
      setBranchesLoading(false)
    }
  }, [repoPath])

  const submit = async () => {
    const branch = name.trim()

    if (pending || !repoPath || !branch) {
      return
    }

    setPending(true)

    try {
      // Pass the typed value as both the dir slug source and the branch, so the
      // branch is exactly what the user named (the dir is slugified git-side).
      const result = await startWorkInRepo(repoPath, { branch, name: branch })

      if (result) {
        onStarted(result.path)
        setOpen(false)
        setName('')
      }
    } catch (err) {
      notifyError(err, p.startWorkFailed)
    } finally {
      setPending(false)
    }
  }

  // Check an EXISTING branch out into a fresh worktree (no new branch).
  const convert = async (branch: HermesGitBranch) => {
    if (pending || !repoPath || !branch) {
      return
    }

    setPending(true)

    try {
      const result = branch.worktreePath
        ? { branch: branch.name, path: branch.worktreePath }
        : await startWorkInRepo(repoPath, { existingBranch: branch.name })

      if (result) {
        onStarted(result.path)
        setOpen(false)
      }
    } catch (err) {
      notifyError(err, p.startWorkFailed)
    } finally {
      setPending(false)
    }
  }

  const enterConvert = () => {
    setConvertMode(true)
    void loadBranches()
  }

  return (
    <>
      <button
        aria-label={p.startWork}
        className="grid size-4 shrink-0 place-items-center rounded-sm bg-transparent text-(--ui-text-quaternary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/section:opacity-100 focus-visible:opacity-100"
        onClick={() => {
          setConvertMode(false)
          setName('')
          setOpen(true)
        }}
        type="button"
      >
        <Codicon name="git-branch" size="0.75rem" />
      </button>
      <Dialog onOpenChange={next => !pending && setOpen(next)} open={open}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{convertMode ? p.convertBranchTitle : p.newWorktreeTitle}</DialogTitle>
            <DialogDescription>{convertMode ? p.convertBranchDesc : p.newWorktreeDesc}</DialogDescription>
          </DialogHeader>

          {convertMode ? (
            <Command
              className="rounded-md border border-(--ui-stroke-tertiary)"
              filter={(value, search) => (value.toLowerCase().includes(search.toLowerCase()) ? 1 : 0)}
            >
              <CommandInput autoFocus disabled={pending} placeholder={p.convertBranchPlaceholder} />
              <CommandList className="max-h-64">
                <CommandEmpty>{branchesLoading ? p.branchesLoading : p.noBranches}</CommandEmpty>
                <CommandGroup>
                  {branches.map(branch => (
                    <CommandItem
                      disabled={pending}
                      key={branch.name}
                      onSelect={() => void convert(branch)}
                      value={branch.name}
                    >
                      <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="git-branch" size="0.8rem" />
                      <span className="truncate">{branch.name}</span>
                      {branch.checkedOut && (
                        <span className="ml-auto shrink-0 text-[0.625rem] text-(--ui-text-tertiary)">
                          {p.branchCheckedOut}
                        </span>
                      )}
                    </CommandItem>
                  ))}
                </CommandGroup>
              </CommandList>
            </Command>
          ) : (
            <SanitizedInput
              autoFocus
              disabled={pending}
              onKeyDown={event => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  void submit()
                } else if (event.key === 'Escape') {
                  setOpen(false)
                }
              }}
              onValueChange={setName}
              placeholder={p.branchPlaceholder}
              sanitize={gitRef}
              value={name}
            />
          )}

          {convertMode ? (
            // The picker is a sub-screen: a single "Cancel" link steps back to
            // the new-branch screen (the dialog's own ✕ / Esc still closes it).
            <DialogFooter className="sm:justify-start">
              <Button
                className="px-0 text-(--ui-text-secondary) hover:text-foreground"
                disabled={pending}
                onClick={() => setConvertMode(false)}
                type="button"
                variant="link"
              >
                {t.common.cancel}
              </Button>
            </DialogFooter>
          ) : (
            <DialogFooter className="sm:justify-between">
              {/* Switch into the convert-an-existing-branch picker. */}
              <Button
                className="px-0 text-(--ui-text-secondary) hover:text-foreground"
                disabled={pending}
                onClick={enterConvert}
                type="button"
                variant="link"
              >
                {p.convertBranchInstead}
              </Button>
              <div className="flex items-center gap-2">
                <Button disabled={pending} onClick={() => setOpen(false)} type="button" variant="ghost">
                  {t.common.cancel}
                </Button>
                <Button disabled={pending || !name.trim()} onClick={() => void submit()} type="button">
                  {p.startWork}
                </Button>
              </div>
            </DialogFooter>
          )}
        </DialogContent>
      </Dialog>
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
