import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import type { HermesGitBaseBranch } from '@/global'
import { useI18n } from '@/i18n'
import { $repoStatus } from '@/store/coding-status'
import { listBaseBranches } from '@/store/projects'

// Filterable combobox for picking the base branch of a new worktree. Lists
// local + remote-tracking branches, defaults to the default branch
// (origin/HEAD, or local main/master when no remote). The current session's
// branch is sorted to the top so it's one click away. The parent owns the
// selected value via `value` / `onValueChange`.
export function BaseBranchPicker({
  disabled,
  repoPath,
  onValueChange,
  value
}: {
  disabled?: boolean
  repoPath: string
  onValueChange: (value: string) => void
  value: string
}) {
  const { t } = useI18n()
  const p = t.sidebar.projects
  const repoStatus = useStore($repoStatus)
  const [branches, setBranches] = useState<HermesGitBaseBranch[]>([])
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(false)

  const currentBranch = repoStatus?.detached ? null : (repoStatus?.branch ?? null)

  const load = useCallback(async () => {
    if (!repoPath) {
      return
    }

    setLoading(true)

    try {
      const list = await listBaseBranches(repoPath)
      setBranches(list)

      // Default to the remote default (origin/HEAD). Fall back to the local
      // default branch (main/master) when no remote exists. The value is
      // always a concrete branch — never undefined.
      const defaultBranch = list.find(b => b.isDefault)

      if (defaultBranch) {
        onValueChange(defaultBranch.name)
      } else {
        onValueChange(list[0]?.name ?? '')
      }
    } catch {
      setBranches([])
    } finally {
      setLoading(false)
    }
  }, [repoPath, onValueChange])

  // Load on mount so the default branch fills in before the user opens the
  // popover — otherwise the button reads "branch off " with nothing after it.
  useEffect(() => {
    if (branches.length === 0 && !loading) {
      void load()
    }
  }, [branches.length, loading, load])

  // Pin the current session's branch to the top, keep the rest in git's
  // most-recently-committed order.
  const sorted = useMemo(() => {
    if (!currentBranch) {
      return branches
    }

    const idx = branches.findIndex(b => b.name === currentBranch)

    if (idx <= 0) {
      return branches
    }

    return [branches[idx], ...branches.slice(0, idx), ...branches.slice(idx + 1)]
  }, [branches, currentBranch])

  // The i18n function returns { before, after } so the branch name can be
  // wrapped in its own styled (underlined) span — works for any word order.
  const parts = p.branchOff()

  return (
    <div className="space-y-1.5">
      <Popover
        onOpenChange={next => {
          if (next && branches.length === 0 && !loading) {
            void load()
          }

          setOpen(next)
        }}
        open={open}
      >
        <PopoverTrigger asChild>
          <Button
            className="group w-full flex justify-start items-center min-w-0 gap-1.5 hover:no-underline hover:text-muted-foreground"
            disabled={disabled || loading}
            size="inline"
            variant="text"
          >
            <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="git-branch" size="0.8rem" />
            <span className="shrink-0">{parts.before}</span>
            <span className="shrink-0 text-primary underline-offset-4 decoration-current/20 group-hover:underline">
              {loading ? '...' : value}
            </span>
            <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="chevron-down" size="0.75rem" />
            <span className="shrink-0">{parts.after}</span>
          </Button>
        </PopoverTrigger>
        <PopoverContent align="start" className="z-[140] min-w-(--radix-popover-trigger-width) p-0">
          <Command filter={(searchValue, search) => (searchValue.toLowerCase().includes(search.toLowerCase()) ? 1 : 0)}>
            <CommandInput autoFocus placeholder={p.baseBranchPlaceholder} />
            <CommandList className="max-h-64">
              <CommandEmpty>{p.baseBranchNone}</CommandEmpty>
              <CommandGroup>
                {sorted.map(branch => (
                  <CommandItem
                    key={branch.name}
                    onSelect={() => {
                      onValueChange(branch.name)
                      setOpen(false)
                    }}
                    value={branch.name}
                  >
                    <div className="flex items-center justify-start gap-1.5">
                      <Codicon
                        className="shrink-0 text-(--ui-text-tertiary)"
                        name={branch.isRemote ? 'repo' : 'git-branch'}
                        size="0.8rem"
                      />
                      {branch.isDefault && (
                        <span className="ml-auto shrink-0 text-[0.625rem] text-(--ui-text-tertiary)">★</span>
                      )}
                      <span className="truncate">{branch.name}</span>
                      {value === branch.name && (
                        <Codicon className="ml-auto shrink-0 text-(--ui-accent)" name="check" size="0.8rem" />
                      )}
                    </div>
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  )
}
