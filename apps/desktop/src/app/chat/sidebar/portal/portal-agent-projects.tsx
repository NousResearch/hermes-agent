import { useStore } from '@nanostores/react'
import { type ReactNode, useState } from 'react'

import { requestComposerFocus } from '@/app/chat/composer/focus'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { ChevronRight, MoreVertical, Pencil, Plus, Trash2 } from '@/lib/icons'
import { useStoreSelector } from '@/lib/use-session-slice'
import type { AgentSidebarOwner } from '@/store/agent-sidebar-preferences'
import { activeGatewayConnectionId } from '@/store/gateway'
import { notify } from '@/store/notifications'
import {
  $pendingPortalProjectForNewChat,
  $portalProjectExpanded,
  armPortalProjectNewChat,
  PORTAL_PROJECT_NAME_MAX,
  type PortalProject,
  setPortalProjectExpanded
} from '@/store/portal-projects'
import {
  addPortalProjectFolderSynced,
  deletePortalProjectSynced,
  renamePortalProjectSynced,
  setPortalProjectAppearanceSynced
} from '@/store/portal-projects-sync'
import { pickProjectFolder, requestStartWorkSession, revealPath } from '@/store/projects'
import { $selectedStoredSessionId } from '@/store/session'

import { ProjectAppearancePicker } from '../projects/project-appearance'
import { startSimplifiedAgentSession } from '../simplified-new-session-model'
import type { SimplifiedAgent } from '../use-simplified-agents'
interface ProjectRowProps {
  agent: SimplifiedAgent
  allAgentSessionIds: readonly string[]
  owner: AgentSidebarOwner
  project: PortalProject
  sessions: SessionInfo[]
  open: boolean
  renderChats: (sessions: SessionInfo[]) => ReactNode
}

function PortalProjectRow({
  agent,
  allAgentSessionIds,
  open,
  owner,
  project,
  renderChats,
  sessions
}: ProjectRowProps) {
  const { t } = useI18n()
  const hasDraftHere = useStoreSelector($pendingPortalProjectForNewChat, pending => pending?.projectId === project.id)
  const [truncated, setTruncated] = useState(false)
  const [renaming, setRenaming] = useState(false)
  const [draft, setDraft] = useState(project.name)
  const iconName = project.icon || 'folder'
  const iconStyle = project.color ? { color: project.color } : undefined

  const checkTruncation = (node: HTMLSpanElement | null) => {
    if (node) {
      setTruncated(node.scrollWidth > node.clientWidth + 1)
    }
  }

  const startChatInProject = () => {
    armPortalProjectNewChat(owner, project.id, allAgentSessionIds, $selectedStoredSessionId.get())

    if (project.path) {
      requestStartWorkSession(project.path, undefined, {
        owner: {
          connectionId: owner.connectionId ?? activeGatewayConnectionId() ?? 'local',
          profile: agent.profile,
          ...(agent.targetProfile ? { targetProfile: agent.targetProfile } : {})
        }
      })
    } else {
      startSimplifiedAgentSession(agent)
    }

    requestComposerFocus()
  }

  const attachFolder = async () => {
    try {
      const picked = await pickProjectFolder()

      if (picked) {
        addPortalProjectFolderSynced(owner, project, picked)
        notify({ durationMs: 3_000, kind: 'success', message: `Attached folder to ${project.name}.` })
      }
    } catch {
      // Picker cancelled or unavailable.
    }
  }

  const commitRename = () => {
    const trimmed = draft.trim()

    if (trimmed && trimmed.length <= PORTAL_PROJECT_NAME_MAX && trimmed !== project.name) {
      renamePortalProjectSynced(owner, project, trimmed)
    }

    setRenaming(false)
  }

  const labelSpan = (
    <span className="portal-project-name" onMouseEnter={event => checkTruncation(event.currentTarget)}>
      {project.name}
    </span>
  )
  const rowTooltip = project.path ? `${project.name} — ${project.path}` : project.name
  return (
    <div className="portal-project-group" data-open={open || undefined} data-slot="portal-project-group">
      <div className="portal-project-item">
        {renaming ? (
          <div className="portal-project-row is-renaming">
            <Codicon aria-hidden className="portal-project-icon" name={iconName} size="0.875rem" style={iconStyle} />
            <input
              aria-label={`Rename ${project.name}`}
              autoFocus
              className="portal-project-rename-input"
              maxLength={PORTAL_PROJECT_NAME_MAX}
              onBlur={commitRename}
              onChange={event => setDraft(event.target.value)}
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
            <Codicon aria-hidden className="portal-project-icon" name={iconName} size="0.875rem" style={iconStyle} />
            {truncated || project.path ? (
              <Tip label={rowTooltip} side="right">
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
          <div className="portal-project-actions">
            <Tip label={`New chat in ${project.name}`}>
              <Button
                aria-label={`New chat in ${project.name}`}
                className="portal-project-action-btn"
                onClick={event => {
                  event.stopPropagation()
                  startChatInProject()
                }}
                size="icon-xs"
                variant="ghost"
              >
                <Plus aria-hidden />
              </Button>
            </Tip>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  aria-label={`Project actions: ${project.name}`}
                  className="portal-project-action-btn"
                  onClick={event => event.stopPropagation()}
                  size="icon-xs"
                  variant="ghost"
                >
                  <MoreVertical aria-hidden />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onSelect={startChatInProject}>
                  <Plus aria-hidden className="size-3.5" />
                  New chat
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                {project.path ? (
                  <DropdownMenuItem onSelect={() => void revealPath(project.path ?? null)}>
                    <Codicon aria-hidden name="folder-opened" size="0.875rem" />
                    Reveal folder
                  </DropdownMenuItem>
                ) : null}
                <DropdownMenuItem onSelect={() => void attachFolder()}>
                  <Codicon aria-hidden name="new-folder" size="0.875rem" />
                  {project.path ? 'Add folder…' : 'Attach folder…'}
                </DropdownMenuItem>
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger>
                    <Codicon aria-hidden name="symbol-color" size="0.875rem" />
                    {t.sidebar.projects.menuAppearance}
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent className="w-48 p-2.5">
                    <ProjectAppearancePicker
                      color={project.color ?? null}
                      icon={project.icon ?? null}
                      noColorLabel={t.sidebar.projects.noColor}
                      onColor={color => setPortalProjectAppearanceSynced(owner, project, { color })}
                      onIcon={icon => setPortalProjectAppearanceSynced(owner, project, { icon })}
                    />
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
                <DropdownMenuSeparator />
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
                    deletePortalProjectSynced(owner, project)
                    notify({ durationMs: 3_000, kind: 'info', message: `${project.name} deleted.` })
                  }}
                  variant="destructive"
                >
                  <Trash2 aria-hidden className="size-3.5" />
                  Delete project
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        )}
      </div>

      {open && (
        <div className="portal-project-chats">
          {hasDraftHere && (
            <div className="portal-session-item">
              <div
                aria-current="page"
                className="portal-session-row"
                data-slot="portal-project-draft-chat"
                onClick={() => requestComposerFocus()}
              >
                <span aria-hidden className="portal-row-mark" data-slot="session-status" data-state="idle">
                  <span>
                    <span />
                  </span>
                </span>
                <span className="truncate">New chat</span>
              </div>
            </div>
          )}
          {sessions.length > 0 ? (
            renderChats(sessions)
          ) : !hasDraftHere ? (
            <button className="portal-project-empty-new" onClick={startChatInProject} type="button">
              <Plus aria-hidden className="size-3.5" />
              <span>New chat</span>
            </button>
          ) : null}
        </div>
      )}
    </div>
  )
}

export interface PortalAgentProjectsProps {
  agent: SimplifiedAgent
  allAgentSessionIds: readonly string[]
  owner: AgentSidebarOwner
  projects: readonly PortalProject[]
  sessionsByProject: ReadonlyMap<string, SessionInfo[]>
  searching: boolean
  renderChats: (sessions: SessionInfo[]) => ReactNode
}

/** Projects nested inside an expanded agent block, above its loose chats (Figma 1701:9935). */
export function PortalAgentProjects({
  agent,
  allAgentSessionIds,
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
            agent={agent}
            allAgentSessionIds={allAgentSessionIds}
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
