import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuRadioGroup, DropdownMenuRadioItem, dropdownMenuRow, DropdownMenuSearch, dropdownMenuSectionLabel, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { ErrorState } from '@/components/ui/error-state'
import { Loader } from '@/components/ui/loader'
import { useI18n } from '@/i18n'
import { FolderOpen } from '@/lib/icons'
import { type CodingWorkspaceOwner, listCodingWorkspaceProjects } from '@/store/coding-workspaces'
import type { ProjectInfo } from '@/types/hermes'

interface CodingProjectPickerProps {
  owner: CodingWorkspaceOwner
  path?: string
  disabled?: boolean
  onSelect: (project: ProjectInfo | null) => void
  onBrowse: () => void
}

export function CodingProjectPicker({ owner, path, disabled, onSelect, onBrowse }: CodingProjectPickerProps) {
  const { t } = useI18n()
  const c = t.codingWorkspace
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')

  const { data, isPending, error, refetch } = useQuery({
    queryKey: ['coding-project-picker', owner.connectionId, owner.profile],
    enabled: open,
    queryFn: () => listCodingWorkspaceProjects(owner),
    staleTime: 0
  })

  const projects = (data ?? []).filter(p => !p.archived && p.primary_path)
  const needle = search.trim().toLocaleLowerCase()
  const matches = projects.filter(p => `${p.name} ${p.primary_path}`.toLocaleLowerCase().includes(needle))
  const selected = projects.find(p => p.primary_path === path)
  const label = selected?.name ?? path?.replace(/[\\/]+$/, '').split(/[\\/]/).pop() ?? c.noProject

  const choose = (value: string) => {
    if (disabled || value === (path ?? '')) {return}
    onSelect(projects.find(project => project.primary_path === value) ?? null)
  }

  return <DropdownMenu onOpenChange={setOpen} open={open}>
    <DropdownMenuTrigger asChild>
      <Button aria-label={`${c.project}: ${label}`} className="min-w-0 shrink" disabled={disabled} size="inline" type="button" variant="text">
        <span className="max-w-48 truncate font-normal">{label}</span><Codicon name="chevron-down" />
      </Button>
    </DropdownMenuTrigger>
    <DropdownMenuContent align="start" aria-label={c.project} className="w-72 max-w-[calc(100vw-2rem)] p-0" side="top">
      <DropdownMenuLabel className={dropdownMenuSectionLabel}>{c.project}</DropdownMenuLabel>
      {projects.length > 0 && <DropdownMenuSearch aria-label={c.searchProjects} onValueChange={setSearch} placeholder={c.searchProjects} value={search} />}
      <DropdownMenuRadioGroup onValueChange={choose} value={path ?? ''}>
        <DropdownMenuRadioItem className={dropdownMenuRow} disabled={disabled} value="">{c.noProject}</DropdownMenuRadioItem>
        {matches.map(project => <DropdownMenuRadioItem className={dropdownMenuRow} disabled={disabled} key={project.id} value={project.primary_path!}>
          <span className="min-w-0"><span className="block truncate">{project.name}</span><span className="block truncate text-(--ui-text-tertiary)" title={project.primary_path!}>{project.primary_path}</span></span>
        </DropdownMenuRadioItem>)}
      </DropdownMenuRadioGroup>
      {isPending && <Loader label={c.project} />}
      {error && <ErrorState description={<span>{String(error.message)}</span>} title={<span>{c.projectsFailed}</span>}>
        <DropdownMenuItem className={dropdownMenuRow} disabled={disabled} onSelect={event => { event.preventDefault(); void refetch() }}>{t.common.retry}</DropdownMenuItem>
      </ErrorState>}
      {!isPending && !error && matches.length === 0 && <DropdownMenuLabel className={dropdownMenuSectionLabel}>{c.noProjects}</DropdownMenuLabel>}
      <DropdownMenuSeparator />
      <DropdownMenuItem className={dropdownMenuRow} disabled={disabled} onSelect={onBrowse}><FolderOpen />{c.browse}</DropdownMenuItem>
    </DropdownMenuContent>
  </DropdownMenu>
}
