import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $gitBranches,
  checkoutBranch,
  createBranch,
  deleteBranch,
  fetchRemote,
  refreshBranches
} from '@/store/git'

interface BranchPickerProps {
  current: string | null
  busy: boolean
}

/** A branch dropdown: current branch button → list of local/remote branches
 *  with checkout, plus create-branch and delete actions. */
export function BranchPicker({ current, busy }: BranchPickerProps) {
  const { t } = useI18n()
  const g = t.sourceControl
  const branches = useStore($gitBranches)
  const [open, setOpen] = useState(false)
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [filter, setFilter] = useState('')
  const rootRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    void refreshBranches()

    const onDown = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false)
        setCreating(false)
      }
    }

    document.addEventListener('mousedown', onDown)

    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const onCheckout = async (name: string) => {
    setOpen(false)
    await checkoutBranch(name)
  }

  const onCreate = async () => {
    const name = newName.trim()

    if (!name) {
      return
    }

    setCreating(false)
    setNewName('')
    setOpen(false)
    await createBranch(name)
  }

  const q = filter.trim().toLowerCase()
  const localMatches = branches.local.filter(b => !q || b.name.toLowerCase().includes(q))
  const remoteMatches = branches.remote.filter(b => !q || b.name.toLowerCase().includes(q))

  return (
    <div className="relative" ref={rootRef}>
      <button
        aria-label={g.switchBranch}
        className="flex h-6 max-w-[11rem] items-center gap-1 rounded-sm px-1.5 text-[0.72rem] font-medium text-foreground hover:bg-(--ui-control-hover-background)"
        disabled={busy}
        onClick={() => setOpen(o => !o)}
        type="button"
      >
        <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="git-branch" size="0.85rem" />
        <span className="min-w-0 truncate">{current ?? g.noBranch}</span>
        <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="chevron-down" size="0.7rem" />
      </button>

      {open && (
        <div className="absolute left-0 top-7 z-50 max-h-80 w-64 overflow-hidden rounded-md border border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background) shadow-lg">
          <div className="flex items-center gap-1 border-b border-(--ui-stroke-tertiary) px-1.5 py-1">
            <Codicon className="text-(--ui-text-tertiary)" name="search" size="0.75rem" />
            <input
              autoFocus
              className="h-5 min-w-0 flex-1 bg-transparent text-[0.72rem] text-foreground outline-none placeholder:text-muted-foreground/50"
              onChange={event => setFilter(event.target.value)}
              placeholder={g.filterBranches}
              value={filter}
            />
          </div>

          <div className="max-h-52 overflow-y-auto py-1">
            {localMatches.map(b => (
              <BranchRow
                branch={b}
                deleteLabel={g.deleteBranch}
                key={`l:${b.name}`}
                onCheckout={() => void onCheckout(b.name)}
                onDelete={
                  b.current ? undefined : async () => { await deleteBranch(b.name, true) }
                }
              />
            ))}

            {remoteMatches.length > 0 && (
              <div className="px-2 pb-0.5 pt-1.5 text-[0.6rem] font-semibold uppercase tracking-wide text-muted-foreground/55">
                {g.remoteBranches}
              </div>
            )}
            {remoteMatches.map(b => (
              <BranchRow
                branch={b}
                key={`r:${b.name}`}
                onCheckout={() => void onCheckout(b.name)}
                remote
              />
            ))}
          </div>

          <div className="border-t border-(--ui-stroke-tertiary) p-1">
            {creating ? (
              <div className="flex items-center gap-1 px-1">
                <input
                  autoFocus
                  className="h-6 min-w-0 flex-1 rounded-sm border border-(--ui-stroke-quaternary) bg-(--ui-bg-secondary) px-1.5 text-[0.72rem] text-foreground outline-none"
                  onChange={event => setNewName(event.target.value)}
                  onKeyDown={event => {
                    if (event.key === 'Enter') {
                      event.preventDefault()
                      void onCreate()
                    } else if (event.key === 'Escape') {
                      setCreating(false)
                    }
                  }}
                  placeholder={g.newBranchName}
                  value={newName}
                />
                <button
                  aria-label={g.create}
                  className="grid size-6 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
                  onClick={() => void onCreate()}
                  type="button"
                >
                  <Codicon name="check" size="0.8rem" />
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-1">
                <button
                  className="flex h-6 flex-1 items-center gap-1.5 rounded-sm px-1.5 text-[0.72rem] text-foreground hover:bg-(--ui-control-hover-background)"
                  onClick={() => setCreating(true)}
                  type="button"
                >
                  <Codicon name="add" size="0.8rem" />
                  {g.createBranch}
                </button>
                <button
                  aria-label={g.fetch}
                  className="grid size-6 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
                  onClick={() => void fetchRemote()}
                  type="button"
                >
                  <Codicon name="sync" size="0.8rem" />
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

interface BranchRowProps {
  branch: { name: string; current: boolean; ahead: number; behind: number }
  remote?: boolean
  deleteLabel?: string
  onCheckout: () => void
  onDelete?: () => void
}

function BranchRow({ branch, remote, deleteLabel, onCheckout, onDelete }: BranchRowProps) {
  return (
    <div className="group/br flex h-6 items-center gap-1.5 px-2 hover:bg-(--ui-control-hover-background)">
      <Codicon
        className={cn('shrink-0', branch.current ? 'text-emerald-500' : 'text-(--ui-text-tertiary)')}
        name={branch.current ? 'check' : remote ? 'cloud' : 'git-branch'}
        size="0.75rem"
      />
      <button
        className="min-w-0 flex-1 truncate text-left text-[0.72rem] text-foreground"
        onClick={onCheckout}
        type="button"
      >
        {branch.name}
      </button>
      {(branch.ahead > 0 || branch.behind > 0) && (
        <span className="shrink-0 text-[0.6rem] text-muted-foreground/55">
          {branch.ahead > 0 ? `↑${branch.ahead}` : ''}
          {branch.behind > 0 ? `↓${branch.behind}` : ''}
        </span>
      )}
      {onDelete && (
        <button
          aria-label={deleteLabel}
          className="grid size-4 shrink-0 place-items-center rounded-sm text-(--ui-text-tertiary) opacity-0 hover:bg-(--ui-bg-secondary) hover:text-destructive group-hover/br:opacity-100"
          onClick={onDelete}
          type="button"
        >
          <Codicon name="trash" size="0.7rem" />
        </button>
      )}
    </div>
  )
}
