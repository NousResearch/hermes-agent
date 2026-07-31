import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { SanitizedInput } from '@/components/ui/sanitized-input'
import type { HermesGitBranch } from '@/global'
import { useI18n } from '@/i18n'
import { gitRef } from '@/lib/sanitize'
import { notifyError } from '@/store/notifications'
import {
  $projectTree,
  $worktreeDialog,
  closeWorktreeDialog,
  listRepoBranches,
  projectRootCwd,
  requestStartWorkSession,
  startWorkInRepo,
  switchBranchInRepo
} from '@/store/projects'

import { BaseBranchPicker } from './base-branch-picker'

interface BranchActionCopy {
  branchCreateWorktree: string
  branchOpenExisting: string
  branchSwitchHome: string
}

const branchActionLabel = (branch: HermesGitBranch, copy: BranchActionCopy) => {
  if (branch.checkedOut) {
    return copy.branchOpenExisting
  }

  return branch.isDefault ? copy.branchSwitchHome : copy.branchCreateWorktree
}

/**
 * The "new worktree" dialog. It is mounted exactly once, in the sidebar beside
 * ProjectDialog, and the `$worktreeDialog` atom drives it. Every entry point
 * (⌘⇧B, the kebab of the coding rail, the + button of the sidebar) publishes
 * its intent to that atom. No entry point mounts its own copy. N composers on
 * screen gave N stacked dialogs for one keypress.
 *
 * Features:
 * - Project picker: change the repo before you name the branch
 * - Branch name input, made safe as a git ref
 * - Base branch picker: a combobox with a filter
 * - Convert mode: check an existing branch out into a worktree
 */
export function WorktreeDialog() {
  const { t } = useI18n()
  const p = t.sidebar.projects
  const state = useStore($worktreeDialog)
  const open = state !== null
  const projectTree = useStore($projectTree)

  const [name, setName] = useState('')
  const [pending, setPending] = useState(false)
  const [convertMode, setConvertMode] = useState(false)
  const [branches, setBranches] = useState<HermesGitBranch[]>([])
  const [branchesLoading, setBranchesLoading] = useState(false)
  const [selectedBase, setSelectedBase] = useState('')
  // The repo that the dialog targets. It is seeded from the resolved intent.
  // This component then owns it, so the project picker can change the target
  // and the user does not reopen the dialog.
  const [repoPath, setRepoPath] = useState('')
  const [projectOpen, setProjectOpen] = useState(false)

  // Every project with a working root is a valid target. The list is deduped by
  // path, because an auto project and a user project can share one folder.
  const projectOptions = useMemo(() => {
    const seen = new Set<string>()

    return projectTree.flatMap(node => {
      const path = projectRootCwd(node)

      if (!path || seen.has(path)) {
        return []
      }

      seen.add(path)

      return [{ label: node.label, path }]
    })
  }, [projectTree])

  const activeProjectLabel =
    projectOptions.find(option => option.path === repoPath)?.label ?? repoPath.split('/').pop() ?? repoPath

  // Reset to a fresh state each time the dialog opens. Apply the resolved repo
  // and the base branch that the caller selected, for example "branch off from
  // main" in the dropdown menu of the coding row.
  useEffect(() => {
    if (state) {
      setName('')
      setConvertMode(false)
      setSelectedBase(state.base ?? '')
      setRepoPath(state.repoPath)
      setBranches([])
    }
  }, [state])

  const onOpenChange = (next: boolean) => {
    if (!next && !pending) {
      closeWorktreeDialog()
    }
  }

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

  // Give the new worktree to a fresh session, then close the dialog.
  const started = (path: string) => {
    requestStartWorkSession(path)
    closeWorktreeDialog()
  }

  const submit = async () => {
    const branch = name.trim()

    if (pending || !repoPath || !branch) {
      return
    }

    setPending(true)

    try {
      const result = await startWorkInRepo(repoPath, { base: selectedBase || undefined, branch, name: branch })

      if (result) {
        started(result.path)
        setName('')
      }
    } catch (err) {
      notifyError(err, p.startWorkFailed)
    } finally {
      setPending(false)
    }
  }

  const convert = async (branch: HermesGitBranch) => {
    if (pending || !repoPath || !branch) {
      return
    }

    setPending(true)

    try {
      let result: null | { branch: string; path: string }

      if (branch.worktreePath) {
        result = { branch: branch.name, path: branch.worktreePath }
      } else if (branch.isDefault) {
        await switchBranchInRepo(repoPath, branch.name)
        result = { branch: branch.name, path: repoPath }
      } else {
        result = await startWorkInRepo(repoPath, { existingBranch: branch.name })
      }

      if (result) {
        started(result.path)
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
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{convertMode ? p.convertBranchTitle : p.newWorktreeTitle}</DialogTitle>
          <DialogDescription>{convertMode ? p.convertBranchDesc : p.newWorktreeDesc}</DialogDescription>
        </DialogHeader>

        {/* Project picker: change the repo that the worktree is cut from. Show
            it only when there is another project to select. */}
        {projectOptions.length > 1 && (
          <Popover onOpenChange={setProjectOpen} open={projectOpen}>
            <PopoverTrigger asChild>
              <Button
                className="group w-full flex justify-start items-center min-w-0 gap-1.5 hover:no-underline hover:text-muted-foreground"
                disabled={pending}
                size="inline"
                variant="text"
              >
                <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="folder" size="0.8rem" />
                <span className="shrink-0">{p.worktreeProjectLabel}</span>
                <span className="truncate text-primary underline-offset-4 decoration-current/20 group-hover:underline">
                  {activeProjectLabel}
                </span>
                <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="chevron-down" size="0.75rem" />
              </Button>
            </PopoverTrigger>
            <PopoverContent align="start" className="z-(--z-modal-popover) min-w-(--radix-popover-trigger-width) p-0">
              <Command
                filter={(value, search) => (value.toLowerCase().includes(search.toLowerCase()) ? 1 : 0)}
              >
                <CommandInput autoFocus placeholder={p.worktreeProjectPlaceholder} />
                <CommandList className="max-h-64">
                  <CommandEmpty>{p.worktreeProjectNone}</CommandEmpty>
                  <CommandGroup>
                    {projectOptions.map(option => (
                      <CommandItem
                        key={option.path}
                        onSelect={() => {
                          setRepoPath(option.path)
                          // The new repo has its own branches. Drop the old
                          // list and the old base, so nothing stale stays.
                          setBranches([])
                          setSelectedBase('')
                          setProjectOpen(false)
                        }}
                        value={`${option.label} ${option.path}`}
                      >
                        <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="repo" size="0.8rem" />
                        <span className="truncate">{option.label}</span>
                        {option.path === repoPath && (
                          <Codicon className="ml-auto shrink-0 text-(--ui-accent)" name="check" size="0.8rem" />
                        )}
                      </CommandItem>
                    ))}
                  </CommandGroup>
                </CommandList>
              </Command>
            </PopoverContent>
          </Popover>
        )}

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
                    <span className="ml-auto shrink-0 text-[0.625rem] text-(--ui-text-tertiary)">
                      {branchActionLabel(branch, p)}
                    </span>
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        ) : (
          <>
            <SanitizedInput
              autoFocus
              disabled={pending}
              onKeyDown={event => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  void submit()
                } else if (event.key === 'Escape') {
                  onOpenChange(false)
                }
              }}
              onValueChange={setName}
              placeholder={p.branchPlaceholder}
              sanitize={gitRef}
              value={name}
            />
            <BaseBranchPicker
              disabled={pending}
              // Remount on a repo change, so the picker loads the branches of
              // the new repo and does not show those of the previous project.
              key={repoPath}
              onValueChange={setSelectedBase}
              repoPath={repoPath}
              value={selectedBase}
            />
          </>
        )}

        {convertMode ? (
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
              <Button disabled={pending} onClick={() => onOpenChange(false)} type="button" variant="ghost">
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
  )
}
