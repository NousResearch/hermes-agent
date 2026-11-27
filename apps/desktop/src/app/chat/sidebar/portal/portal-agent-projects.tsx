import { useStore } from '@nanostores/react'
import { type ReactNode, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { ChevronRight, MoreVertical, Pencil, Trash2 } from '@/lib/icons'
import type { AgentSidebarOwner } from '@/store/agent-sidebar-preferences'
import { notify } from '@/store/notifications'
import {
  $portalProjectExpanded,
  deletePortalProject,
  PORTAL_PROJECT_NAME_MAX,
  type PortalProject,
  renamePortalProject,
  setPortalProjectExpanded
} from '@/store/portal-projects'

interface ProjectRowProps {
  owner: AgentSidebarOwner
  project: PortalProject
  sessions: SessionInfo[]
  open: boolean
  renderChats: (sessions: SessionInfo[]) => ReactNode
}

function PortalProjectRow({ open, owner, project, renderChats, sessions }: ProjectRowProps) {
  const [truncated, setTruncated] = useState(false)
  const [renaming, setRenaming] = useState(false)
  const [draft, setDraft] = useState(project.name)

  const checkTruncation = (node: HTMLSpanElement | null) => {
    if (node) {
      setTruncated(node.scrollWidth > node.clientWidth + 1)
    }
  }

  const commitRename = () => {
    const trimmed = draft.trim()

    if (trimmed && trimmed.length <= PORTAL_PROJECT_NAME_MAX && trimmed !== project.name) {
      renamePortalProject(owner, project.id, trimmed)
    }

    setRenaming(false)
  }

  const labelSpan = (
    <span
      className="portal-project-name"
      onMouseEnter={event => checkTruncation(event.currentTarget)}
    >
      {project.name}
    </span>
  )

  return (
    <div className="portal-project-group" data-open={open || undefined} data-slot="portal-project-group">
      <div className="portal-project-item">
        {renaming ? (
          <div className="portal-project-row is-renaming">
            <Codicon aria-hidden className="portal-project-icon" name="folder" size="0.875rem" />
            <input
              aria-label={`Rename ${project.name}`}
              autoFocus
              className="portal-project-rename-input"
              maxLength={PORTAL_PROJECT_NAME_MAX}
              onBlur={commitRename}
              onKeyDown={event => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  commitRename()
                } else if (event.key === 'Escape') {
                  event.preventDefault()
                  setDraft(project.name)
                  setRenaming(false)
                }
              }}
              type="text"
              value={draft}
            />
          </div>
        ) : (
          <button
            aria-expanded={open}
            className="portal-project-row"
            data-slot="portal-project-row"
            onClick={() => setPortalProjectExpanded(project.id, !open)}
            onFocus={event => checkTruncation(event.currentTarget.querySelector('.portal-project-name'))}
            type="button"
          >
            <Codicon aria-hidden className="portal-project-icon" name="folder" size="0.875rem" />
            {truncated ? (
              <Tip label={project.name} side="right">
                {labelSpan}
              </Tip>
            ) : (
              labelSpan
            )}
            <span className="portal-project-count">{sessions.length}</span>
            <ChevronRight aria-hidden className="portal-project-chevron" />
          </button>
        )}

        {!renaming && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                aria-label={`Project actions: ${project.name}`}
                className="portal-row-action portal-project-action"
                onClick={event => event.stopPropagation()}
                size="icon-xs"
                variant="ghost"
              >
                <MoreVertical aria-hidden />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem
                onSelect={() => {
                  setDraft(project.name)
                  setRenaming(true)
                }}
              >
                <Pencil aria-hidden className="size-3.5" />
                Rename
              </DropdownMenuItem>
              <DropdownMenuItem
                className="text-destructive focus:text-destructive"
                onSelect={() => {
                  deletePortalProject(owner, project.id)
                  notify({ durationMs: 3_000, kind: 'info', message: `${project.name} deleted.` })
                }}
                variant="destructive"
              >
                <Trash2 aria-hidden className="size-3.5" />
                Delete project
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      {open && sessions.length > 0 && <div className="portal-project-chats">{renderChats(sessions)}</div>}
    </div>
  )
}

export interface PortalAgentProjectsProps {
  owner: AgentSidebarOwner
  projects: readonly PortalProject[]
  sessionsByProject: ReadonlyMap<string, SessionInfo[]>
  searching: boolean
  renderChats: (sessions: SessionInfo[]) => ReactNode
}

/** Projects nested inside an expanded agent block, above its loose chats (Figma 1701:9935). */
export function PortalAgentProjects({
  owner,
  projects,
  renderChats,
  searching,
  sessionsByProject
}: PortalAgentProjectsProps) {
  const expanded = useStore($portalProjectExpanded)

  if (projects.length === 0) {
    return null
  }

  return (
    <div className="portal-agent-projects" data-slot="portal-agent-projects">
      {projects.map(project => {
        const chats = sessionsByProject.get(project.id) ?? []

        if (searching && chats.length === 0) {
          return null
        }

        const open = searching || (expanded[project.id] ?? false)

        return (
          <PortalProjectRow
            key={project.id}
            open={open}
            owner={owner}
            project={project}
            renderChats={renderChats}
            sessions={chats}
          />
        )
      })}
    </div>
  )
}
