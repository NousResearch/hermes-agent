import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Command,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator
} from '@/components/ui/command'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, ChevronDown, ChevronLeft, FolderOpen, Loader2, Plus } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import {
  $recentProjects,
  filterRecentProjects,
  isProjectPickerEnabled,
  mergeRecentProjects,
  normalizeProjectPath,
  projectDisplayName,
  recordRecentProject,
  sessionProjectEntries
} from '@/store/recent-projects'
import { $connection, $sessions, getConfiguredDefaultProjectDir } from '@/store/session'

import { isProjectCwd } from '../sidebar/workspace-groups'

interface ProjectPickerProps {
  cwd?: null | string
  disabled?: boolean
  /** Bind the chosen directory as the session cwd — the existing
   *  changeSessionCwd (new-session sets $currentCwd; live session RPC). */
  onChangeCwd: (cwd: string) => Promise<void> | void
}

type PickerView = 'create' | 'list'

/** New-conversation project picker (hc-517). A "project" is just the session
 *  cwd/git-root the sidebar already groups by — this chip lets the user choose
 *  one from a searchable MRU list, browse to an existing folder, or scaffold a
 *  new blank folder, before the first message pins it via session.create. */
export function ProjectPicker({ cwd, disabled, onChangeCwd }: ProjectPickerProps) {
  const { t } = useI18n()
  const copy = t.composer.projectPicker
  const connection = useStore($connection)
  const persisted = useStore($recentProjects)
  const sessions = useStore($sessions)

  const [open, setOpen] = useState(false)
  const [view, setView] = useState<PickerView>('list')
  const [query, setQuery] = useState('')
  const [newName, setNewName] = useState('')
  const [newParent, setNewParent] = useState('')
  const [busy, setBusy] = useState(false)
  const nameInputRef = useRef<HTMLInputElement | null>(null)

  const desktop = typeof window === 'undefined' ? undefined : window.hermesDesktop
  const canCreateBlank = typeof desktop?.createProjectDir === 'function'

  const projects = useMemo(
    () => mergeRecentProjects(persisted, sessionProjectEntries(sessions)),
    [persisted, sessions]
  )

  const filtered = useMemo(() => filterRecentProjects(projects, query), [projects, query])

  const currentPath = normalizeProjectPath(cwd ?? '')
  const currentIsProject = isProjectCwd(currentPath)
  const chipLabel = currentIsProject ? projectDisplayName(currentPath) : copy.select

  const resetTransient = useCallback(() => {
    setView('list')
    setQuery('')
    setNewName('')
    setNewParent('')
    setBusy(false)
  }, [])

  const handleOpenChange = useCallback(
    (next: boolean) => {
      setOpen(next)

      if (!next) {
        resetTransient()
      }
    },
    [resetTransient]
  )

  const bindProject = useCallback(
    async (path: string, name?: string) => {
      const target = path.trim()

      if (!target) {
        return
      }

      recordRecentProject(target, name)
      triggerHaptic('selection')
      handleOpenChange(false)
      await onChangeCwd(target)
    },
    [handleOpenChange, onChangeCwd]
  )

  const defaultDialogPath = useCallback(
    () => newParent.trim() || getConfiguredDefaultProjectDir() || currentPath || undefined,
    [currentPath, newParent]
  )

  const openExistingFolder = useCallback(async () => {
    if (!desktop?.selectPaths) {
      return
    }

    setBusy(true)

    try {
      const picked = await desktop.selectPaths({
        directories: true,
        multiple: false,
        title: copy.useExistingTitle,
        defaultPath: defaultDialogPath()
      })

      const dir = picked?.[0]?.trim()

      if (dir) {
        await bindProject(dir)
      }
    } catch (err) {
      notifyError(err, copy.pickFailed)
    } finally {
      setBusy(false)
    }
  }, [bindProject, copy.pickFailed, copy.useExistingTitle, defaultDialogPath, desktop])

  const chooseParent = useCallback(async () => {
    if (!desktop?.selectPaths) {
      return
    }

    try {
      const picked = await desktop.selectPaths({
        directories: true,
        multiple: false,
        title: copy.chooseParentTitle,
        defaultPath: defaultDialogPath()
      })

      const dir = picked?.[0]?.trim()

      if (dir) {
        setNewParent(dir)
      }
    } catch (err) {
      notifyError(err, copy.pickFailed)
    }
  }, [copy.chooseParentTitle, copy.pickFailed, defaultDialogPath, desktop])

  const createBlankProject = useCallback(async () => {
    const name = newName.trim()
    const parent = newParent.trim()

    if (!desktop?.createProjectDir || !name || !parent || busy) {
      return
    }

    setBusy(true)

    try {
      const result = await desktop.createProjectDir(parent, name)

      if (result?.ok && result.path) {
        await bindProject(result.path, name)
      } else {
        notifyError(new Error(result?.error || copy.createFailed), copy.createFailed)
      }
    } catch (err) {
      notifyError(err, copy.createFailed)
    } finally {
      setBusy(false)
    }
  }, [bindProject, busy, copy.createFailed, desktop, newName, newParent])

  const enterCreateView = useCallback(() => {
    setNewParent(prev => prev || getConfiguredDefaultProjectDir() || '')
    setView('create')
  }, [])

  // Focus the name field when the create view opens.
  useEffect(() => {
    if (open && view === 'create') {
      const id = window.requestAnimationFrame(() => nameInputRef.current?.focus())

      return () => window.cancelAnimationFrame(id)
    }

    return undefined
  }, [open, view])

  // Gate: feature flag off, non-Electron, or a remote backend (its cwd lives on
  // the remote host — the local folder dialog can't reach it). Fall back to the
  // plain new-conversation flow with no chip.
  if (!isProjectPickerEnabled() || !desktop || connection?.mode === 'remote') {
    return null
  }

  const createValid = newName.trim().length > 0 && newParent.trim().length > 0

  return (
    <div className="flex min-w-0 items-center">
      <Popover onOpenChange={handleOpenChange} open={open}>
        <PopoverTrigger asChild>
          <Button
            aria-label={copy.label}
            className={cn(
              'h-6 max-w-full gap-1 rounded-full border border-(--ui-stroke-secondary) bg-transparent px-2 text-[0.72rem] font-medium text-muted-foreground',
              'hover:border-(--ui-stroke-primary) hover:bg-(--chrome-action-hover) hover:text-foreground',
              'data-[state=open]:border-(--ui-stroke-primary) data-[state=open]:text-foreground',
              currentIsProject && 'text-foreground'
            )}
            disabled={disabled}
            size="sm"
            title={currentIsProject ? currentPath : copy.select}
            type="button"
            variant="ghost"
          >
            <FolderOpen className="size-3 shrink-0" />
            <span className="min-w-0 truncate">{chipLabel}</span>
            <ChevronDown className="size-3 shrink-0 opacity-70" />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="start" className="w-80 p-0" side="top" sideOffset={8}>
          {view === 'list' ? (
            <Command shouldFilter={false}>
              <CommandInput onValueChange={setQuery} placeholder={copy.searchPlaceholder} value={query} />
              <CommandList>
                {/* Plain empty row, not cmdk's <CommandEmpty> — the always-
                    present action items below keep cmdk's match count > 0, so
                    its count-gated Empty would never show. */}
                {filtered.length === 0 && (
                  <div className="px-2 py-6 text-center text-sm text-muted-foreground">
                    {projects.length === 0 ? copy.noRecent : copy.noMatches}
                  </div>
                )}
                {filtered.length > 0 && (
                  <CommandGroup heading={copy.recentHeading}>
                    {filtered.map(project => (
                      <CommandItem
                        key={project.path}
                        onSelect={() => void bindProject(project.path, project.name)}
                        value={project.path}
                      >
                        <FolderOpen className="size-4 shrink-0 text-muted-foreground" />
                        <span className="flex min-w-0 flex-col">
                          <span className="truncate text-sm text-foreground">{project.name}</span>
                          <span className="truncate text-[0.68rem] text-muted-foreground/80">{project.path}</span>
                        </span>
                        {normalizeProjectPath(project.path) === currentPath && (
                          <Check className="ml-auto size-4 shrink-0 text-foreground" />
                        )}
                      </CommandItem>
                    ))}
                  </CommandGroup>
                )}
                <CommandSeparator />
                <CommandGroup>
                  <CommandItem
                    disabled={busy}
                    onSelect={() => void openExistingFolder()}
                    value="__project_picker_use_existing__"
                  >
                    <FolderOpen className="size-4 shrink-0 text-muted-foreground" />
                    <span className="text-sm">{copy.useExisting}</span>
                  </CommandItem>
                  {canCreateBlank && (
                    <CommandItem onSelect={enterCreateView} value="__project_picker_new_blank__">
                      <Plus className="size-4 shrink-0 text-muted-foreground" />
                      <span className="text-sm">{copy.newBlank}</span>
                    </CommandItem>
                  )}
                </CommandGroup>
              </CommandList>
            </Command>
          ) : (
            <div className="grid gap-3 p-3">
              <div className="flex items-center gap-1.5">
                <Button
                  aria-label={copy.back}
                  className="size-6"
                  onClick={() => setView('list')}
                  size="icon"
                  type="button"
                  variant="ghost"
                >
                  <ChevronLeft className="size-4" />
                </Button>
                <span className="text-sm font-medium text-foreground">{copy.newTitle}</span>
              </div>

              <Input
                onChange={event => setNewName(event.target.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter' && createValid && !busy) {
                    event.preventDefault()
                    void createBlankProject()
                  }
                }}
                placeholder={copy.namePlaceholder}
                ref={nameInputRef}
                value={newName}
              />

              <div className="grid gap-1">
                <span className="text-[0.7rem] font-medium uppercase tracking-wide text-muted-foreground/85">
                  {copy.locationLabel}
                </span>
                <button
                  className="flex items-center gap-2 rounded-md border border-(--ui-stroke-secondary) px-2 py-1.5 text-left text-[0.75rem] text-muted-foreground transition-colors hover:border-(--ui-stroke-primary) hover:text-foreground"
                  onClick={() => void chooseParent()}
                  type="button"
                >
                  <FolderOpen className="size-3.5 shrink-0" />
                  <span className="min-w-0 flex-1 truncate">{newParent || copy.chooseParent}</span>
                </button>
              </div>

              <div className="flex justify-end gap-1.5">
                <Button onClick={() => setView('list')} size="sm" type="button" variant="ghost">
                  {t.common.cancel}
                </Button>
                <Button disabled={!createValid || busy} onClick={() => void createBlankProject()} size="sm" type="button">
                  {busy ? <Loader2 className="size-3.5 animate-spin" /> : null}
                  <span>{copy.create}</span>
                </Button>
              </div>
            </div>
          )}
        </PopoverContent>
      </Popover>
    </div>
  )
}
