import { useStore } from '@nanostores/react'
import { memo, useCallback, useEffect, useRef, useState } from 'react'

import { StatusRow } from '@/components/chat/status-row'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList
} from '@/components/ui/command'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { DiffCount } from '@/components/ui/diff-count'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { SanitizedInput } from '@/components/ui/sanitized-input'
import type { HermesGitBranch } from '@/global'
import { useI18n } from '@/i18n'
import { gitRef } from '@/lib/sanitize'
import { $repoStatus, $repoWorktrees } from '@/store/coding-status'
import { notifyError } from '@/store/notifications'
import { $newWorktreeRequest } from '@/store/projects'

// Tiny uppercase section header, matching the composer "+" menu's labels.
const MENU_SECTION = 'text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)'

interface CodingStatusRowProps {
  /** Branch the current draft off into a fresh worktree + session, based on
   *  `base` (a branch name; omitted = current HEAD). The composer owns the
   *  draft, so it supplies the orchestration; the row just collects the new
   *  branch name + base. Omitted (e.g. remote backend) hides the affordance. */
  onBranchOff?: (branch: string, base?: string) => Promise<void>
  /** Check an existing branch out into a fresh worktree + session (no new
   *  branch). Drives the dialog's "convert a branch" picker. */
  onConvertBranch?: (branch: string, path?: null | string) => Promise<void>
  /** List the repo's local branches for the "convert a branch" picker. */
  onListBranches?: () => Promise<HermesGitBranch[]>
  /** Open the review pane (changed files + diffs). */
  onOpen?: () => void
  /** Jump into an existing worktree (open a fresh session anchored there). */
  onOpenWorktree?: (path: string) => void
  /** Switch the current repo checkout to another branch. */
  onSwitchBranch?: (branch: string) => Promise<void>
}

/**
 * The always-on coding-context row, the BASE of the composer status stack:
 * current branch, dirty summary (+/-), and ahead/behind. A touch more prominent
 * than the per-turn rows above it (larger branch label, accent glyph), and the
 * entry point to the review pane. Hidden when the active session isn't in a
 * local git repo (the probe returns null).
 */
export const CodingStatusRow = memo(function CodingStatusRow({
  onBranchOff,
  onConvertBranch,
  onListBranches,
  onOpen,
  onOpenWorktree,
  onSwitchBranch
}: CodingStatusRowProps) {
  const { t } = useI18n()
  const s = t.statusStack.coding
  const p = t.sidebar.projects
  const status = useStore($repoStatus)
  const worktrees = useStore($repoWorktrees)

  const [branchOpen, setBranchOpen] = useState(false)
  const [branchName, setBranchName] = useState('')
  const [branchBase, setBranchBase] = useState<string | undefined>(undefined)
  const [branchPending, setBranchPending] = useState(false)
  // "Convert an existing branch into a worktree" sub-mode of the dialog: the body
  // swaps the new-branch name input for a filterable list of the repo's branches.
  const [convertMode, setConvertMode] = useState(false)
  const [branches, setBranches] = useState<HermesGitBranch[]>([])
  const [branchesLoading, setBranchesLoading] = useState(false)

  // Pull the repo's branches the first time the convert picker is shown for an
  // open dialog. Cheap + bounded; refreshed each time the picker is entered so a
  // branch created mid-session shows up.
  const loadBranches = useCallback(async () => {
    if (!onListBranches) {
      return
    }

    setBranchesLoading(true)

    try {
      setBranches(await onListBranches())
    } catch {
      setBranches([])
    } finally {
      setBranchesLoading(false)
    }
  }, [onListBranches])

  // Open the name dialog for a chosen base. Deferred so the dropdown finishes
  // closing before the dialog grabs focus (Radix focus-trap handoff races
  // otherwise).
  const startBranch = (base: string | undefined) => {
    setBranchBase(base)
    setBranchName('')
    setConvertMode(false)
    setTimeout(() => setBranchOpen(true), 0)
  }

  // Open the dialog straight into the convert-a-branch picker.
  const startConvert = () => {
    setBranchBase(undefined)
    setBranchName('')
    setConvertMode(true)
    void loadBranches()
    setTimeout(() => setBranchOpen(true), 0)
  }

  // Flip an already-open dialog into the picker (the in-dialog link).
  const enterConvert = () => {
    setConvertMode(true)
    void loadBranches()
  }

  const convertBranch = async (branch: HermesGitBranch) => {
    if (branchPending || !branch || !onConvertBranch) {
      return
    }

    setBranchPending(true)

    try {
      await onConvertBranch(branch.name, branch.worktreePath)
      setBranchOpen(false)
    } catch (err) {
      notifyError(err, p.startWorkFailed)
    } finally {
      setBranchPending(false)
    }
  }

  // Global ⌘⇧B (workspace.newWorktree): open the name dialog for a worktree off
  // current HEAD. The rail only renders inside a repo, so the hotkey naturally
  // no-ops elsewhere. Guarded by a token ref so it fires on the keypress, not on
  // mount or unrelated re-renders.
  const worktreeReq = useStore($newWorktreeRequest)
  const lastWorktreeReqRef = useRef(worktreeReq)

  useEffect(() => {
    if (worktreeReq === lastWorktreeReqRef.current) {
      return
    }

    lastWorktreeReqRef.current = worktreeReq

    if (!onBranchOff) {
      return
    }

    setBranchBase(undefined)
    setBranchName('')
    setConvertMode(false)
    setBranchOpen(true)
  }, [onBranchOff, worktreeReq])

  const submitBranch = async () => {
    const branch = branchName.trim()

    if (branchPending || !branch || !onBranchOff) {
      return
    }

    setBranchPending(true)

    try {
      await onBranchOff(branch, branchBase)
      setBranchOpen(false)
      setBranchName('')
    } catch (err) {
      notifyError(err, p.startWorkFailed)
    } finally {
      setBranchPending(false)
    }
  }

  const switchToBranch = async (branch: string) => {
    if (!onSwitchBranch) {
      return
    }

    try {
      await onSwitchBranch(branch)
    } catch (err) {
      notifyError(err, s.switchFailed(branch))
    }
  }

  if (!status) {
    return null
  }

  const branchLabel = status.detached ? s.detached : status.branch || s.noBranch
  // The kebab offers branching off the trunk and/or the current branch. The
  // worktree-add bases the new branch on `base` (a branch name; undefined =
  // current HEAD). We dedupe so "on main" shows a single trunk entry, and fall
  // back to a plain off-HEAD branch when no trunk is detected.
  const current = status.detached ? null : status.branch
  const branchTargets: { base: string | undefined; label: string }[] = []

  // Current branch first (the 99% "branch off where I am"), then the trunk just
  // below it ("New branch from main"), deduped when they're the same.
  if (current) {
    branchTargets.push({ base: current, label: s.branchOffFrom(current) })
  }

  if (status.defaultBranch && status.defaultBranch !== current) {
    branchTargets.push({ base: status.defaultBranch, label: s.branchOffFrom(status.defaultBranch) })
  }

  if (branchTargets.length === 0) {
    branchTargets.push({ base: undefined, label: s.newBranch })
  }

  const switchTarget = onSwitchBranch && current && status.defaultBranch && status.defaultBranch !== current ? status.defaultBranch : null

  // Other worktrees to jump into — everything except the one we're already in
  // (matched by its checked-out branch) and the bare/main placeholder entry.
  const otherWorktrees = onOpenWorktree
    ? worktrees.filter(w => w.path && !w.detached && w.branch && w.branch !== current)
    : []

  const hasLineDelta = status.added > 0 || status.removed > 0
  // Untracked files carry no line delta vs HEAD, so surface them as a count when
  // they're the only change (otherwise +/- tells the story).
  const untrackedOnly = !hasLineDelta && status.untracked > 0

  return (
    <>
      <StatusRow
        // The base "where am I working" strip is part of the composer surface
        // itself, so it inherits the composer's width and clipped top radius.
        className="coding-status-bar min-h-7 rounded-t-[inherit] rounded-b-none border-b border-(--ui-stroke-tertiary) px-3.5 py-1.5 hover:bg-transparent"
        // Static branch glyph — never the loading spinner. This row only renders
        // once `status` exists, so a spinner here only ever fired on *refreshes*
        // of an already-loaded repo (window focus, turn settle), reading as an
        // annoying icon "blip" with no first-load value. Refreshes are silent.
        leading={<Codicon className="text-(--ui-green)" name="git-branch" size="0.8rem" />}
        onActivate={onOpen}
      >
        <div className="flex min-w-0 flex-1 items-center gap-1">
          <span
            className="min-w-0 truncate text-xs font-normal text-muted-foreground/92 transition-colors group-hover/status-row:text-foreground/90"
            title={branchLabel}
          >
            {branchLabel}
          </span>

          {/* Branch actions kebab — same pattern as the session/worktree rows.
              ALWAYS laid out; only its opacity flips on hover/focus/open, so
              revealing it never reflows the row (no layout shift). pointer-events
              follow opacity so the invisible trigger isn't clickable at rest. */}
          {onBranchOff && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  aria-label={s.newBranch}
                  className="pointer-events-none size-4 shrink-0 text-muted-foreground/60 opacity-0 transition hover:text-foreground group-hover/status-row:pointer-events-auto group-hover/status-row:opacity-100 group-focus-within/status-row:pointer-events-auto group-focus-within/status-row:opacity-100 data-[state=open]:pointer-events-auto data-[state=open]:opacity-100"
                  onClick={event => event.stopPropagation()}
                  onKeyDown={event => {
                    // The row's onActivate also fires on Enter/Space; keep it from
                    // opening the review pane when the kebab is the focus target.
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.stopPropagation()
                    }
                  }}
                  size="icon-xs"
                  variant="ghost"
                >
                  <Codicon name="kebab-vertical" size="0.8rem" />
                </Button>
              </DropdownMenuTrigger>
              {/* The row sits at the bottom of the screen (above the composer),
                  so the menu opens upward. */}
              <DropdownMenuContent align="end" className="w-60" side="top" sideOffset={6}>
                <DropdownMenuLabel className={MENU_SECTION}>{s.newBranch}</DropdownMenuLabel>
                {branchTargets.map(target => (
                  <DropdownMenuItem key={target.base ?? '__head__'} onSelect={() => startBranch(target.base)}>
                    <span className="truncate">{target.label}</span>
                  </DropdownMenuItem>
                ))}

                {switchTarget && (
                  <DropdownMenuItem onSelect={() => void switchToBranch(switchTarget)}>
                    <span className="truncate">{s.switchTo(switchTarget)}</span>
                  </DropdownMenuItem>
                )}

                <DropdownMenuSeparator />
                <DropdownMenuLabel className={MENU_SECTION}>{s.worktrees}</DropdownMenuLabel>
                {otherWorktrees.map(worktree => (
                  <DropdownMenuItem key={worktree.path} onSelect={() => onOpenWorktree?.(worktree.path)}>
                    <span className="truncate">{worktree.branch}</span>
                  </DropdownMenuItem>
                ))}
                {/* Create a fresh worktree off the current HEAD (the generic
                    "spin up a worktree here", mirroring the sidebar's + button). */}
                <DropdownMenuItem onSelect={() => startBranch(undefined)}>
                  <span className="truncate">{p.startWork}</span>
                </DropdownMenuItem>
                {/* Check an EXISTING branch out into a worktree (no new branch). */}
                {onConvertBranch && (
                  <DropdownMenuItem onSelect={() => startConvert()}>
                    <span className="truncate">{p.convertBranch}</span>
                  </DropdownMenuItem>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>

        {(status.ahead > 0 || status.behind > 0) && (
          <span className="ml-auto flex shrink-0 items-center gap-1.5 text-[0.68rem] leading-4 text-muted-foreground/75 tabular-nums">
            {status.ahead > 0 && (
              <span className="flex items-center gap-0.5" title={s.ahead(status.ahead)}>
                <span aria-hidden>↑</span>
                {status.ahead}
              </span>
            )}
            {status.behind > 0 && (
              <span className="flex items-center gap-0.5" title={s.behind(status.behind)}>
                <span aria-hidden>↓</span>
                {status.behind}
              </span>
            )}
          </span>
        )}

        {hasLineDelta ? (
          <DiffCount
            added={status.added}
            className={`text-[0.72rem] leading-4 ${status.ahead === 0 && status.behind === 0 ? 'ml-auto' : ''}`}
            removed={status.removed}
          />
        ) : untrackedOnly ? (
          <span
            className={`shrink-0 text-[0.72rem] leading-4 text-amber-500/90 ${status.ahead === 0 && status.behind === 0 ? 'ml-auto' : ''}`}
          >
            {s.changed(status.untracked)}
          </span>
        ) : null}
      </StatusRow>

      <Dialog onOpenChange={open => !branchPending && setBranchOpen(open)} open={branchOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{convertMode ? p.convertBranchTitle : p.newWorktreeTitle}</DialogTitle>
            <DialogDescription>
              {convertMode ? p.convertBranchDesc : p.newWorktreeDesc}
              {!convertMode && branchBase && (
                <span className="mt-1 block text-(--ui-text-secondary)">{s.branchOffFrom(branchBase)}</span>
              )}
            </DialogDescription>
          </DialogHeader>

          {convertMode ? (
            <Command
              className="rounded-md border border-(--ui-stroke-tertiary)"
              // The branch name is the authoritative key; filter on it directly.
              filter={(value, search) => (value.toLowerCase().includes(search.toLowerCase()) ? 1 : 0)}
            >
              <CommandInput autoFocus disabled={branchPending} placeholder={p.convertBranchPlaceholder} />
              <CommandList className="max-h-64">
                <CommandEmpty>{branchesLoading ? p.branchesLoading : p.noBranches}</CommandEmpty>
                <CommandGroup>
                  {branches.map(branch => (
                    <CommandItem
                      disabled={branchPending}
                      key={branch.name}
                      onSelect={() => void convertBranch(branch)}
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
              disabled={branchPending}
              onKeyDown={event => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  void submitBranch()
                } else if (event.key === 'Escape') {
                  setBranchOpen(false)
                }
              }}
              onValueChange={setBranchName}
              placeholder={p.branchPlaceholder}
              sanitize={gitRef}
              value={branchName}
            />
          )}

          {convertMode ? (
            // The picker is a sub-screen: a single "Cancel" link steps back to
            // the new-branch screen (the dialog's own ✕ / Esc still closes it).
            <DialogFooter className="sm:justify-start">
              <Button
                className="px-0 text-(--ui-text-secondary) hover:text-foreground"
                disabled={branchPending}
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
              {onConvertBranch ? (
                <Button
                  className="px-0 text-(--ui-text-secondary) hover:text-foreground"
                  disabled={branchPending}
                  onClick={enterConvert}
                  type="button"
                  variant="link"
                >
                  {p.convertBranchInstead}
                </Button>
              ) : (
                <span />
              )}
              <div className="flex items-center gap-2">
                <Button disabled={branchPending} onClick={() => setBranchOpen(false)} type="button" variant="ghost">
                  {t.common.cancel}
                </Button>
                <Button
                  disabled={branchPending || !branchName.trim()}
                  onClick={() => void submitBranch()}
                  type="button"
                >
                  {p.startWork}
                </Button>
              </div>
            </DialogFooter>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
})
