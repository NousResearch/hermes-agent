import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import { cn } from '@/lib/utils'
import { $sidebarProjectsOpen, setSidebarProjectsOpen } from '@/store/layout'
import type { Project } from '@/store/projects'

import { SidebarPanelLabel } from '../../shell/sidebar-label'

interface ProjectsSidebarSectionProps {
  projects: Project[]
  selectedProjectId: string | null
  onSelectProject: (id: string) => void
  onNewProject: () => void
}

export function ProjectsSidebarSection({
  projects,
  selectedProjectId,
  onSelectProject,
  onNewProject
}: ProjectsSidebarSectionProps) {
  const open = useStore($sidebarProjectsOpen)

  return (
    <SidebarGroup className="shrink-0 p-0 pb-1">
      <div className="group/section flex shrink-0 items-center justify-between pb-1 pt-1.5">
        <button
          className="group/section-label flex w-fit items-center gap-1 bg-transparent text-left leading-none"
          onClick={() => setSidebarProjectsOpen(!open)}
          type="button"
        >
          <SidebarPanelLabel>Projects</SidebarPanelLabel>
          <span className="text-[0.6875rem] font-medium text-(--ui-text-quaternary)">{projects.length}</span>
          <DisclosureCaret
            className="text-(--ui-text-tertiary) opacity-0 transition group-hover/section-label:opacity-100"
            open={open}
          />
        </button>
        <Button
          className="h-5 shrink-0 px-1.5 text-[0.6875rem] text-(--ui-text-tertiary) opacity-0 transition hover:text-foreground group-hover/section:opacity-100"
          onClick={event => {
            event.stopPropagation()
            onNewProject()
          }}
          size="sm"
          variant="ghost"
        >
          + New
        </Button>
      </div>
      {open && (
        <SidebarGroupContent className="flex max-h-72 flex-col gap-px overflow-x-hidden overflow-y-auto overscroll-contain pb-1.75 compact:max-h-none compact:overflow-visible">
          {projects.length === 0 ? (
            <div className="px-2 py-1 text-[0.6875rem] text-(--ui-text-tertiary)">No projects yet</div>
          ) : (
            projects.map(project => (
              <ProjectSidebarRow
                isActive={selectedProjectId === project.id}
                key={project.id}
                onClick={() => onSelectProject(project.id)}
                project={project}
              />
            ))
          )}
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}

function ProjectSidebarRow({
  isActive,
  onClick,
  project
}: {
  isActive: boolean
  onClick: () => void
  project: Project
}) {
  return (
    <button
      className={cn(
        'group/project relative flex min-h-[1.625rem] w-full items-center gap-1.5 rounded-md py-0.5 pl-2 pr-1 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40',
        isActive ? 'bg-(--ui-row-active-background) text-foreground' : 'hover:bg-(--chrome-action-hover)'
      )}
      onClick={onClick}
      type="button"
    >
      <span className="grid w-3.5 shrink-0 place-items-center">
        <Codicon
          className={cn('text-(--ui-text-tertiary)', isActive && 'text-foreground')}
          name={isActive ? 'folder-opened' : 'folder'}
          size="0.75rem"
        />
      </span>
      <span className="flex min-w-0 flex-1 flex-col">
        <span
          className={cn(
            'truncate text-[0.8125rem] font-medium leading-snug',
            isActive ? 'text-foreground' : 'text-(--ui-text-secondary) group-hover/project:text-foreground'
          )}
        >
          {project.title}
        </span>
        {project.description && (
          <span className="truncate text-[0.6875rem] leading-snug text-(--ui-text-tertiary)">{project.description}</span>
        )}
      </span>
    </button>
  )
}
