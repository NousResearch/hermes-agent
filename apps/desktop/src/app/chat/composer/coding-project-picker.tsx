import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import { Loader } from '@/components/ui/loader'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { SearchField } from '@/components/ui/search-field'
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

  const choose = (project: ProjectInfo | null) => { setOpen(false); onSelect(project) }

  return <Popover onOpenChange={setOpen} open={open}>
    <PopoverTrigger asChild>
      <Button disabled={disabled} size="micro" type="button" variant="ghost">
        <FolderOpen />{c.project}{': '}<span className="max-w-48 truncate">{selected?.name ?? path ?? c.noProject}</span>
      </Button>
    </PopoverTrigger>
    <PopoverContent align="start" className="max-w-[calc(100vw-2rem)]" side="top">
      {projects.length > 0 && <SearchField aria-label={c.searchProjects} onChange={setSearch} placeholder={c.searchProjects} value={search} />}
      <div className="grid max-h-64 gap-1 overflow-y-auto">
        <Button className="justify-start" onClick={() => choose(null)} size="sm" type="button" variant="ghost">{c.noProject}</Button>
        {isPending && <Loader label={c.project} />}
        {error && <ErrorState description={<span>{String(error.message)}</span>} title={<span>{c.projectsFailed}</span>}>
          <Button onClick={() => void refetch()} size="micro" type="button" variant="ghost">{t.common.retry}</Button>
        </ErrorState>}
        {matches.map(project => <Button className="justify-start" key={project.id} onClick={() => choose(project)} size="sm" type="button" variant="ghost">
          <span className="min-w-0 text-left"><span className="block truncate">{project.name}</span><span className="block truncate text-(--ui-text-tertiary)">{project.primary_path}</span></span>
        </Button>)}
        {!isPending && !error && matches.length === 0 && <span className="text-xs text-(--ui-text-tertiary)">{c.noProjects}</span>}
        <Button className="justify-start" onClick={() => { setOpen(false); onBrowse() }} size="sm" type="button" variant="ghost"><FolderOpen />{c.browse}</Button>
      </div>
    </PopoverContent>
  </Popover>
}
