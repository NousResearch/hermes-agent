import { atom } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'
import { agentSidebarKey, type AgentSidebarOwner } from '@/store/agent-sidebar-preferences'

/** New project names are capped at 25 characters so they fit cleanly in the sidebar (Figma 1715:9972).
 *  Existing names longer than 25 characters remain intact and truncate with a hover tooltip (1719:9980). */
export const PORTAL_PROJECT_NAME_MAX = 25

export interface PortalProject {
  id: string
  name: string
  createdAt: number
}

export interface AgentPortalProjects {
  projects: PortalProject[]
  /** Session ID -> project ID membership for this agent. */
  sessionProject: Record<string, string>
}

export interface PortalCreateProjectRequest {
  owner: AgentSidebarOwner
  agentLabel: string
  /** Present for Flow A (chat menu -> Move to project -> New project…): moves this chat on create. */
  session?: {
    id: string
    title: string
  }
}

const EMPTY_AGENT_PROJECTS: AgentPortalProjects = {
  projects: [],
  sessionProject: {}
}

function isPortalProject(value: unknown): value is PortalProject {
  return (
    Boolean(value) &&
    typeof value === 'object' &&
    typeof (value as PortalProject).id === 'string' &&
    typeof (value as PortalProject).name === 'string' &&
    typeof (value as PortalProject).createdAt === 'number'
  )
}

function parseAgentProjects(value: unknown): AgentPortalProjects | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const raw = value as Record<string, unknown>
  const projects = Array.isArray(raw.projects) ? raw.projects.filter(isPortalProject) : []
  const validIds = new Set(projects.map(project => project.id))
  const sessionProject: Record<string, string> = {}

  if (raw.sessionProject && typeof raw.sessionProject === 'object' && !Array.isArray(raw.sessionProject)) {
    for (const [sessionId, projectId] of Object.entries(raw.sessionProject as Record<string, unknown>)) {
      if (typeof projectId === 'string' && validIds.has(projectId)) {
        sessionProject[sessionId] = projectId
      }
    }
  }

  return { projects, sessionProject }
}

/** Redesign-only per-agent projects (Figma 1698:99887), keyed by `agentSidebarKey`. */
export const $portalAgentProjects = persistentAtom<Record<string, AgentPortalProjects>>(
  'hermes.desktop.portalSidebar.agentProjects.v1',
  {},
  Codecs.json(value => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return {}
    }

    const out: Record<string, AgentPortalProjects> = {}

    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
      const parsed = parseAgentProjects(entry)

      if (parsed && parsed.projects.length > 0) {
        out[key] = parsed
      }
    }

    return out
  })
)

/** Redesign-only expand/collapse state per project ID. */
export const $portalProjectExpanded = persistentAtom<Record<string, boolean>>(
  'hermes.desktop.portalSidebar.projectExpanded.v1',
  {},
  Codecs.json(value =>
    value && typeof value === 'object' && !Array.isArray(value)
      ? Object.fromEntries(Object.entries(value).filter(([, open]) => typeof open === 'boolean'))
      : {}
  )
)

/** Modal state for "Create a new project" (Flow A from chat menu, Flow B from agent menu). */
export const $portalCreateProjectDialog = atom<PortalCreateProjectRequest | null>(null)

export interface PendingPortalProjectNewChat {
  owner: AgentSidebarOwner
  projectId: string
  baselineIds: ReadonlySet<string>
}

/** Armed when the user starts a new chat from inside a project; files the first newly created
 *  session for that agent into the project automatically. */
export const $pendingPortalProjectForNewChat = atom<PendingPortalProjectNewChat | null>(null)

export function armPortalProjectNewChat(
  owner: AgentSidebarOwner,
  projectId: string,
  existingSessionIds: Iterable<string>
): void {
  setPortalProjectExpanded(projectId, true)
  $pendingPortalProjectForNewChat.set({
    baselineIds: new Set(existingSessionIds),
    owner,
    projectId
  })
}

export function clearPendingPortalProjectNewChat(): void {
  if ($pendingPortalProjectForNewChat.get() !== null) {
    $pendingPortalProjectForNewChat.set(null)
  }
}

export function claimPendingPortalProjectSession(owner: AgentSidebarOwner, sessionId: string): boolean {
  const pending = $pendingPortalProjectForNewChat.get()

  if (!pending || agentSidebarKey(pending.owner) !== agentSidebarKey(owner) || pending.baselineIds.has(sessionId)) {
    return false
  }

  moveSessionToPortalProject(pending.owner, sessionId, pending.projectId)
  $pendingPortalProjectForNewChat.set(null)

  return true
}

export function openPortalCreateProjectDialog(request: PortalCreateProjectRequest): void {
  $portalCreateProjectDialog.set(request)
}

export function closePortalCreateProjectDialog(): void {
  $portalCreateProjectDialog.set(null)
}

export function agentPortalProjectsFor(
  all: Record<string, AgentPortalProjects>,
  owner: AgentSidebarOwner
): AgentPortalProjects {
  return all[agentSidebarKey(owner)] ?? EMPTY_AGENT_PROJECTS
}

function updateAgentEntry(owner: AgentSidebarOwner, next: AgentPortalProjects): void {
  const key = agentSidebarKey(owner)
  const current = $portalAgentProjects.get()

  if (next.projects.length === 0) {
    if (!Object.hasOwn(current, key)) {
      return
    }

    const { [key]: _dropped, ...rest } = current
    $portalAgentProjects.set(rest)

    return
  }

  $portalAgentProjects.set({ ...current, [key]: next })
}

export interface CreatePortalProjectResult {
  project: PortalProject
  previousProjectId: string | undefined
}

export function createPortalProject(
  owner: AgentSidebarOwner,
  rawName: string,
  moveSessionId?: string
): CreatePortalProjectResult | null {
  const name = rawName.trim()

  if (!name || name.length > PORTAL_PROJECT_NAME_MAX) {
    return null
  }

  const entry = agentPortalProjectsFor($portalAgentProjects.get(), owner)
  const id = `pp_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`
  const project: PortalProject = { id, name, createdAt: Date.now() }
  const previousProjectId = moveSessionId ? entry.sessionProject[moveSessionId] : undefined
  const sessionProject = moveSessionId ? { ...entry.sessionProject, [moveSessionId]: id } : entry.sessionProject

  updateAgentEntry(owner, {
    projects: [...entry.projects, project],
    sessionProject
  })

  if (moveSessionId) {
    setPortalProjectExpanded(id, true)
  }

  return { project, previousProjectId }
}

export function deletePortalProject(
  owner: AgentSidebarOwner,
  projectId: string,
  restoreMove?: { sessionId: string; previousProjectId?: string }
): void {
  const entry = agentPortalProjectsFor($portalAgentProjects.get(), owner)
  const projects = entry.projects.filter(project => project.id !== projectId)
  const sessionProject: Record<string, string> = {}

  for (const [sessionId, assignedId] of Object.entries(entry.sessionProject)) {
    if (assignedId !== projectId) {
      sessionProject[sessionId] = assignedId
    }
  }

  if (restoreMove?.previousProjectId && projects.some(p => p.id === restoreMove.previousProjectId)) {
    sessionProject[restoreMove.sessionId] = restoreMove.previousProjectId
  }

  updateAgentEntry(owner, { projects, sessionProject })

  const expanded = $portalProjectExpanded.get()

  if (Object.hasOwn(expanded, projectId)) {
    const { [projectId]: _dropped, ...rest } = expanded
    $portalProjectExpanded.set(rest)
  }
}

export function moveSessionToPortalProject(
  owner: AgentSidebarOwner,
  sessionId: string,
  projectId: string | null
): string | undefined {
  const entry = agentPortalProjectsFor($portalAgentProjects.get(), owner)
  const previousProjectId = entry.sessionProject[sessionId]

  if (projectId === null) {
    if (!previousProjectId) {
      return undefined
    }

    const { [sessionId]: _removed, ...rest } = entry.sessionProject
    updateAgentEntry(owner, { ...entry, sessionProject: rest })

    return previousProjectId
  }

  if (!entry.projects.some(project => project.id === projectId)) {
    return previousProjectId
  }

  updateAgentEntry(owner, {
    ...entry,
    sessionProject: { ...entry.sessionProject, [sessionId]: projectId }
  })
  setPortalProjectExpanded(projectId, true)

  return previousProjectId
}

export function renamePortalProject(owner: AgentSidebarOwner, projectId: string, rawName: string): boolean {
  const name = rawName.trim()

  if (!name || name.length > PORTAL_PROJECT_NAME_MAX) {
    return false
  }

  const entry = agentPortalProjectsFor($portalAgentProjects.get(), owner)

  if (!entry.projects.some(project => project.id === projectId)) {
    return false
  }

  updateAgentEntry(owner, {
    ...entry,
    projects: entry.projects.map(project => (project.id === projectId ? { ...project, name } : project))
  })

  return true
}

export function setPortalProjectExpanded(projectId: string, open: boolean): void {
  $portalProjectExpanded.set({ ...$portalProjectExpanded.get(), [projectId]: open })
}

export function rekeyPortalAgentProjects(fromKey: string, toKey?: string): void {
  const current = $portalAgentProjects.get()

  if (!Object.hasOwn(current, fromKey)) {
    return
  }

  const { [fromKey]: entry, ...rest } = current
  $portalAgentProjects.set(toKey === undefined ? rest : { ...rest, [toKey]: entry })
}
