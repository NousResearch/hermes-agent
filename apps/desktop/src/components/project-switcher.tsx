import { useStore } from '@nanostores/react'
import { Dialog as DialogPrimitive } from 'radix-ui'
import { useEffect, useMemo, useState } from 'react'

import { projectTreeCwd, sortProjectsForOverview } from '@/app/chat/sidebar/projects/model'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { useI18n } from '@/i18n'
import { FolderOpen } from '@/lib/icons'
import { $dismissedAutoProjectIds, filterVisibleProjects } from '@/store/layout'
import { $activeProjectId, $projectTree } from '@/store/projects'

interface ProjectSwitcherDialogProps {
  onOpenChange: (open: boolean) => void
  /** Select one backend-authoritative project id. */
  onSelect: (projectId: string) => void
  /** Browse for a folder and upsert it through the Projects backend. */
  onOpenFolder: () => void
  open: boolean
}

/**
 * Focused project picker over the active gateway's `projects.tree` cache.
 *
 * Projects are per profile and per connection; the backend is authoritative
 * for their identity, activity ordering, and paths. Keeping the picker on the
 * shared Projects cache means a profile or local/remote switch replaces the
 * rows at the same boundary as the sidebar instead of leaking paths through a
 * window-global MRU.
 */
export function ProjectSwitcherDialog({ onOpenChange, onOpenFolder, onSelect, open }: ProjectSwitcherDialogProps) {
  const { t } = useI18n()

  const copy = t.projectSwitcher
  const tree = useStore($projectTree)
  const activeProjectId = useStore($activeProjectId)
  const dismissedAutoProjectIds = useStore($dismissedAutoProjectIds)
  const [search, setSearch] = useState('')

  const rows = useMemo(
    () => sortProjectsForOverview(filterVisibleProjects(tree, dismissedAutoProjectIds), activeProjectId),
    [activeProjectId, dismissedAutoProjectIds, tree]
  )

  useEffect(() => {
    if (!open) {
      setSearch('')
    }
  }, [open])

  return (
    <DialogPrimitive.Root onOpenChange={onOpenChange} open={open}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-[200] bg-black/15 backdrop-blur-[1px] data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0" />
        <DialogPrimitive.Content
          aria-describedby={undefined}
          className="fixed left-1/2 top-[14vh] z-[210] w-[min(40rem,calc(100vw-2rem))] -translate-x-1/2 overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-chat-bubble-background) shadow-lg duration-150 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:slide-in-from-top-2 data-[state=open]:zoom-in-95"
        >
          <DialogPrimitive.Title className="sr-only">{copy.title}</DialogPrimitive.Title>
          <Command className="bg-transparent" loop>
            <CommandInput onValueChange={setSearch} placeholder={copy.searchPlaceholder} value={search} />
            <CommandList className="max-h-[min(24rem,60vh)]">
              <CommandEmpty>{copy.empty}</CommandEmpty>
              {rows.length === 0 && search === '' && (
                <div className="px-3 py-6 text-center text-sm text-muted-foreground">{copy.empty}</div>
              )}
              {rows.length > 0 && (
                <CommandGroup
                  className="**:[[cmdk-group-heading]]:uppercase **:[[cmdk-group-heading]]:tracking-wider **:[[cmdk-group-heading]]:text-[0.6875rem] **:[[cmdk-group-heading]]:text-muted-foreground/70"
                  heading={t.commandCenter.projects}
                >
                  {rows.map(project => {
                    const path = projectTreeCwd(project)
                    const isActive = project.id === activeProjectId

                    return (
                      <CommandItem
                        className="gap-2.5"
                        key={project.id}
                        onSelect={() => {
                          onSelect(project.id)
                          onOpenChange(false)
                        }}
                        value={`${project.label} ${path ?? ''}`}
                      >
                        <FolderOpen className="size-4 shrink-0 text-muted-foreground" />
                        <span className="flex min-w-0 flex-col leading-snug">
                          <span className="truncate">{project.label}</span>
                          {path && <span className="truncate text-xs text-muted-foreground/70">{path}</span>}
                        </span>
                        {isActive && <span className="ml-auto size-1.5 shrink-0 rounded-full bg-foreground/70" />}
                      </CommandItem>
                    )
                  })}
                </CommandGroup>
              )}
              <CommandGroup>
                <CommandItem
                  className="gap-2.5"
                  onSelect={() => {
                    onOpenChange(false)
                    onOpenFolder()
                  }}
                  value={copy.openFolder}
                >
                  <FolderOpen className="size-4 shrink-0 text-muted-foreground" />
                  <span className="truncate">{copy.openFolder}</span>
                </CommandItem>
              </CommandGroup>
            </CommandList>
          </Command>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
