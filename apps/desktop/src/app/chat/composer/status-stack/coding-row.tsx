import { useStore } from '@nanostores/react'
import { memo, useEffect, useRef, useState } from 'react'

import { WorktreeDialog } from '@/app/chat/sidebar/projects/worktree-dialog'
import { StatusRow } from '@/components/chat/status-row'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DiffCount } from '@/components/ui/diff-count'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import type { HermesGitBranch } from '@/global'
import { useI18n } from '@/i18n'
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
  onConvertBranch?: (branch: string, path?: null | string, isDefault?: boolean) => Promise<void>
  /** List the repo's local branches for the "convert a branch" picker. */
  onListBranches?: () => Promise<HermesGitBranch[]>
  /** Open the review pane (changed files + diffs). */
  onOpen?: () => void
  /** Jump into an existing worktree (open a fresh session anchored there). */
  onOpenWorktree?: (path: string) => void
  /** Switch the current repo checkout to another branch. */
  onSwitchBranch?: (branch: string) => Promise<void>
  /** Repo root path for the worktree dialog. */
  repoPath?: null | string
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
  onSwitchBranch,
  repoPath
}: CodingStatusRowProps) {
  const { t } = useI18n()
  const s = t.statusStack.coding
  const p = t.sidebar.projects
  const status = useStore($repoStatus)
  const worktrees = useStore($repoWorktrees)

  // Shared worktree dialog — replaces the old inline dialog. Opened by the
  // dropdown menu's "branch off" items and the global ⌘⇧B hotkey.
  const [worktreeOpen, setWorktreeOpen] = useState(false)
  const [worktreeBase, setWorktreeBase] = useState<string | undefined>(undefined)
  const resolvedRepoPath = repoPath?.trim() || undefined

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

  // Global ⌘⇧B (workspace.newWorktree): open the shared worktree dialog. The
  // coding row only renders inside a repo, so the hotkey naturally no-ops
  // elsewhere. Guarded by a token ref so it fires on the keypress, not on
  // mount or unrelated re-renders.
  const worktreeReq = useStore($newWorktreeRequest)
  const lastWorktreeReqRef = useRef(worktreeReq)

  useEffect(() => {
    if (worktreeReq === lastWorktreeReqRef.current) {
      return
    }

    lastWorktreeReqRef.current = worktreeReq

    if (!resolvedRepoPath || !onOpenWorktree) {
      return
    }

    setWorktreeBase(undefined)
    setWorktreeOpen(true)
  }, [onOpenWorktree, resolvedRepoPath, worktreeReq])

  // Open the worktree dialog from the dropdown menu with a pre-selected base.
  const startBranch = (base: string | undefined) => {
    setWorktreeBase(base)
    setTimeout(() => setWorktreeOpen(true), 0)
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

  const switchTarget =
    onSwitchBranch && current && status.defaultBranch && status.defaultBranch !== current ? status.defaultBranch : null

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
                {/* Create a fresh worktree off the current HEAD (the generic
                    "spin up a worktree here", mirroring the sidebar's + button). */}
                {onConvertBranch && (
                  <DropdownMenuItem onSelect={() => startBranch(undefined)}>
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

      {resolvedRepoPath && onOpenWorktree && (
        <WorktreeDialog
          initialBase={worktreeBase}
          onOpenChange={setWorktreeOpen}
          onStarted={onOpenWorktree}
          open={worktreeOpen}
          repoPath={resolvedRepoPath}
        />
      )}
    </>
  )
})
