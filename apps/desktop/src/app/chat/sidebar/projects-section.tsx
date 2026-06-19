import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import type { SessionInfo } from '@/hermes'
import { cn } from '@/lib/utils'
import { $expandedProjectIds, $sidebarProjectsOpen, setSidebarProjectsOpen, toggleExpandedProjectId } from '@/store/layout'
import { removeProject } from '@/store/projects'
import type { Project } from '@/store/projects'
import { sessionPinId } from '@/store/session'

import { SidebarPanelLabel } from '../../shell/sidebar-label'

import { SidebarSessionRow } from './session-row'

interface ProjectsSidebarSectionProps {
  activeSessionId: string | null
  onArchiveSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onNewProject: () => void
  onResumeSession: (sessionId: string) => void
  onSelectProject: (id: string) => void
  onTogglePin: (sessionId: string) => void
  projects: Project[]
  selectedProjectId: string | null
  sessions: SessionInfo[]
}

export function ProjectsSidebarSection({
  activeSessionId,
  onArchiveSession,
  onDeleteSession,
  onNewProject,
  onResumeSession,
  onSelectProject,
  onTogglePin,
  projects,
  selectedProjectId,
  sessions
}: ProjectsSidebarSectionProps) {
  const open = useStore($sidebarProjectsOpen)
  const expandedIds = useStore($expandedProjectIds)

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
            projects.map(project => {
              const projectSessions = sessions.filter(s => s.cwd?.startsWith(project.path))
              const isExpanded = expandedIds.includes(project.id)

              return (
                <div key={project.id}>
                  <ProjectSidebarRow
                    isActive={selectedProjectId === project.id}
                    isExpanded={isExpanded}
                    onClick={() => onSelectProject(project.id)}
                    onToggleExpand={() => toggleExpandedProjectId(project.id)}
                    project={project}
                    sessionCount={projectSessions.length}
                  />
                  {isExpanded && projectSessions.length > 0 && (
                    <div className="ml-4 flex flex-col gap-px">
                      {projectSessions.map(session => (
                        <SidebarSessionRow
                          isPinned={false}
                          isSelected={activeSessionId === session.id}
                          isWorking={false}
                          key={session.id}
                          onArchive={() => onArchiveSession(session.id)}
                          onDelete={() => onDeleteSession(session.id)}
                          onPin={() => onTogglePin(sessionPinId(session))}
                          onResume={() => onResumeSession(session.id)}
                          session={session}
                        />
                      ))}
                    </div>
                  )}
                </div>
              )
            })
          )}
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}

function ProjectSidebarRow({
  isActive,
  isExpanded,
  onClick,
  onToggleExpand,
  project,
  sessionCount
}: {
  isActive: boolean
  isExpanded: boolean
  onClick: () => void
  onToggleExpand: () => void
  project: Project
  sessionCount: number
}) {
  const [deleteOpen, setDeleteOpen] = useState(false)
  const hasSessions = sessionCount > 0

  return (
    <>
      <div
        className={cn(
          'group/project relative flex min-h-[1.625rem] w-full items-center gap-1 rounded-md py-0.5 pl-1 pr-1',
          isActive ? 'bg-(--ui-row-active-background) text-foreground' : 'hover:bg-(--chrome-action-hover)'
        )}
      >
        {/* Per-project collapse toggle — visible when sessions exist */}
        <button
          aria-label={isExpanded ? 'Collapse sessions' : 'Expand sessions'}
          className={cn(
            'grid w-3.5 shrink-0 place-items-center bg-transparent focus-visible:outline-none',
            !hasSessions && 'pointer-events-none opacity-0'
          )}
          onClick={event => {
            event.stopPropagation()
            onToggleExpand()
          }}
          tabIndex={hasSessions ? 0 : -1}
          type="button"
        >
          <DisclosureCaret
            className={cn('text-(--ui-text-tertiary)', isActive && 'text-foreground')}
            open={isExpanded}
          />
        </button>

        <button
          className="flex min-w-0 flex-1 items-center gap-1.5 bg-transparent text-left focus-visible:outline-none"
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
          <span className="flex min-w-0 flex-1 flex-col group-hover/project:pr-5">
            <span
              className={cn(
                'truncate text-[0.8125rem] font-medium leading-snug',
                isActive ? 'text-foreground' : 'text-(--ui-text-secondary) group-hover/project:text-foreground'
              )}
            >
              {project.title}
            </span>
            {project.description && (
              <span className="truncate text-[0.6875rem] leading-snug text-(--ui-text-tertiary)">
                {project.description}
              </span>
            )}
          </span>
          {/* Session count badge — shown when collapsed and sessions exist */}
          {hasSessions && !isExpanded && (
            <span className="mr-5 shrink-0 rounded-full bg-(--ui-control-active-background) px-1.5 py-0.5 text-[0.6rem] font-medium text-(--ui-text-tertiary)">
              {sessionCount}
            </span>
          )}
        </button>

        {/* Delete button — revealed on row hover */}
        <button
          aria-label={`Delete ${project.title}`}
          className="absolute right-1 top-1/2 -translate-y-1/2 rounded p-0.5 text-transparent opacity-0 transition-all hover:bg-(--ui-control-active-background) hover:text-destructive group-hover/project:opacity-100 group-hover/project:text-(--ui-text-tertiary) focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
          onClick={event => {
            event.stopPropagation()
            setDeleteOpen(true)
          }}
          type="button"
        >
          <Codicon name="trash" size="0.75rem" />
        </button>
      </div>

      <ConfirmDialog
        confirmLabel="Delete"
        description={`"${project.title}" will be removed from Hermes. The folder and its files will not be deleted.`}
        destructive
        onClose={() => setDeleteOpen(false)}
        onConfirm={() => {
          removeProject(project.id)
        }}
        open={deleteOpen}
        title={`Delete project?`}
      />
    </>
  )
}

