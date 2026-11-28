import { atom } from 'nanostores'

import { isUnderPath } from '@/lib/path-compare'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { agentSidebarKey, type AgentSidebarOwner } from '@/store/agent-sidebar-preferences'

export interface BackendProjectRecord {
  id: string
  name?: string | null
  slug?: string | null
  primary_path?: string | null
  folders?: readonly { path?: string | null; is_primary?: boolean }[] | null
  color?: string | null
  icon?: string | null
  created_at?: number | null
}
/** New project names are capped at 25 characters so they fit cleanly in the sidebar (Figma 1715:9972).
 *  Existing names longer than 25 characters remain intact and truncate with a hover tooltip (1719:9980). */
export const PORTAL_PROJECT_NAME_MAX = 25

export interface PortalProject {
  id: string
  name: string
  createdAt: number
  /** Primary workspace folder (absent/null for lightweight chat-group projects). */
  path?: string | null
  /** All workspace folders attached to this project (matches Classic `projects.db`). */
  folders?: string[]
  color?: string | null
  icon?: string | null
  /** True when backed by `projects.db` (`projects.create` / `projects.list`). */
  backend?: boolean
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

function sanitizePortalProject(value: unknown): PortalProject | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const raw = value as Record<string, unknown>

  if (typeof raw.id !== 'string' || typeof raw.name !== 'string' || typeof raw.createdAt !== 'number') {
    return null
  }

  const folders = Array.isArray(raw.folders)
    ? raw.folders.filter((entry): entry is string => typeof entry === 'string' && entry.trim().length > 0)
    : undefined

  return {
    id: raw.id,
    name: raw.name,
    createdAt: raw.createdAt,
    ...(typeof raw.path === 'string' && raw.path.trim() ? { path: raw.path.trim() } : {}),
    ...(folders && folders.length > 0 ? { folders } : {}),
    ...(typeof raw.color === 'string' && raw.color.trim() ? { color: raw.color } : {}),
    ...(typeof raw.icon === 'string' && raw.icon.trim() ? { icon: raw.icon } : {}),
    ...(raw.backend === true ? { backend: true } : {})
  }
}
function parseAgentProjects(value: unknown): AgentPortalProjects | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const raw = value as Record<string, unknown>
  const projects = Array.isArray(raw.projects)
    ? raw.projects.map(sanitizePortalProject).filter((project): project is PortalProject => project !== null)
    : []
  const validIds = new Set(projects.map(project => project.id))
  const sessionProject: Record<string, string> = {}

  if (raw.sessionProject && typeof raw.sessionProject === 'object' && !Array.isArray(raw.sessionProject)) {
    for (const [sessionId, projectId] of Object.entries(raw.sessionProject as Record<string, unknown>)) {
      if (typeof projectId === 'string' && (projectId === '' || validIds.has(projectId))) {
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
  /** Session ID that was selected at the instant the user clicked "+" / "New chat" inside the
   *  project, so a render before `startSimplifiedAgentSession` clears selection never disarms it. */
  armedAtSessionId: string | null
  /** Becomes true once selection leaves `armedAtSessionId` (e.g. drops to null for the fresh draft). */
  leftArmedSession: boolean
}

/** Armed when the user starts a new chat from inside a project; files the first newly created
 *  session for that agent into the project automatically. */
export const $pendingPortalProjectForNewChat = atom<PendingPortalProjectNewChat | null>(null)

export function armPortalProjectNewChat(
  owner: AgentSidebarOwner,
  projectId: string,
  existingSessionIds: Iterable<string>,
  armedAtSessionId: string | null = null
): void {
  setPortalProjectExpanded(projectId, true)
  $pendingPortalProjectForNewChat.set({
    armedAtSessionId,
    baselineIds: new Set(existingSessionIds),
    leftArmedSession: armedAtSessionId === null,
    owner,
    projectId
  })
}

export function markPendingPortalProjectLeftArmed(): void {
  const pending = $pendingPortalProjectForNewChat.get()

  if (pending && !pending.leftArmedSession) {
    $pendingPortalProjectForNewChat.set({ ...pending, leftArmedSession: true })
  }
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

/** Sentinel in `sessionProject` meaning the user explicitly detached this session from its
 *  workspace-folder project so `cwd` prefix matching does not silently pull it back in. */
export const DETACHED_PORTAL_PROJECT = ''

/** Resolve which project owns a session: explicit assignment wins (including explicit detach
 *  via `DETACHED_PORTAL_PROJECT`); otherwise longest matching workspace folder prefix (Classic parity). */
export function resolvePortalProjectForSession(
  entry: AgentPortalProjects,
  session: { id: string; cwd?: string | null }
): string | null {
  const explicit = entry.sessionProject[session.id]

  if (explicit === DETACHED_PORTAL_PROJECT) {
    return null
  }

  if (explicit && entry.projects.some(project => project.id === explicit)) {
    return explicit
  }

  const cwd = session.cwd?.trim()

  if (!cwd) {
    return null
  }
  let bestId: string | null = null
  let bestLength = -1

  for (const project of entry.projects) {
    const roots = project.folders?.length ? project.folders : project.path ? [project.path] : []

    for (const root of roots) {
      if (root && isUnderPath(root, cwd) && root.length > bestLength) {
        bestId = project.id
        bestLength = root.length
      }
    }
  }

  return bestId
}

export function projectInfoToPortalProject(project: BackendProjectRecord): PortalProject {
  const folders = (project.folders ?? [])
    .map(folder => folder.path?.trim())
    .filter((folder): folder is string => Boolean(folder))
  const primary = project.primary_path?.trim() || folders[0] || null

  if (primary && !folders.includes(primary)) {
    folders.unshift(primary)
  }

  return {
    backend: true,
    color: project.color ?? null,
    createdAt: project.created_at ?? Date.now(),
    ...(folders.length > 0 ? { folders } : {}),
    icon: project.icon ?? null,
    id: project.id,
    name: project.name || project.id,
    ...(primary ? { path: primary } : {})
  }
}

function projectSignature(project: PortalProject): string {
  return JSON.stringify([
    project.id,
    project.name,
    project.path ?? null,
    project.folders ?? null,
    project.color ?? null,
    project.icon ?? null,
    Boolean(project.backend)
  ])
}

function entriesEqual(a: AgentPortalProjects, b: AgentPortalProjects): boolean {
  if (a.projects.length !== b.projects.length) {
    return false
  }

  for (let i = 0; i < a.projects.length; i += 1) {
    if (projectSignature(a.projects[i]) !== projectSignature(b.projects[i])) {
      return false
    }
  }

  const aKeys = Object.keys(a.sessionProject)
  const bKeys = Object.keys(b.sessionProject)

  if (aKeys.length !== bKeys.length) {
    return false
  }

  return aKeys.every(key => a.sessionProject[key] === b.sessionProject[key])
}

/** Merge authoritative backend projects (`projects.list`) into an agent's cached project list.
 *  Preserves any local-only (`!backend`) fallback projects and valid manual session assignments.
 *  Skips writing `$portalAgentProjects` when nothing semantically changed. */
export function mergeBackendPortalProjects(
  owner: AgentSidebarOwner,
  backendProjects: readonly BackendProjectRecord[]
): void {
  const entry = agentPortalProjectsFor($portalAgentProjects.get(), owner)
  const mapped = backendProjects.map(projectInfoToPortalProject)
  const backendIds = new Set(mapped.map(project => project.id))
  const localOnly = entry.projects.filter(project => !project.backend && !backendIds.has(project.id))
  const merged = [...mapped, ...localOnly]
  const validIds = new Set(merged.map(project => project.id))
  const sessionProject: Record<string, string> = {}

  for (const [sessionId, projectId] of Object.entries(entry.sessionProject)) {
    if (projectId === DETACHED_PORTAL_PROJECT || validIds.has(projectId)) {
      sessionProject[sessionId] = projectId
    }
  }

  const next = { projects: merged, sessionProject }

  if (entriesEqual(entry, next)) {
    return
  }

  updateAgentEntry(owner, next)
}
export function upsertPortalProject(
  owner: AgentSidebarOwner,
  project: PortalProject,
  moveSessionId?: string
): CreatePortalProjectResult {
  const entry = agentPortalProjectsFor($portalAgentProjects.get(), owner)
  const previousProjectId = moveSessionId ? entry.sessionProject[moveSessionId] : undefined
  const exists = entry.projects.some(existing => existing.id === project.id)
  const projects = exists
    ? entry.projects.map(existing => (existing.id === project.id ? project : existing))
    : [...entry.projects, project]
  const sessionProject = moveSessionId ? { ...entry.sessionProject, [moveSessionId]: project.id } : entry.sessionProject

  updateAgentEntry(owner, { projects, sessionProject })

  if (moveSessionId) {
    setPortalProjectExpanded(project.id, true)
  }

  return { previousProjectId, project }
}

export function patchPortalProject(
  owner: AgentSidebarOwner,
  projectId: string,
  patch: Partial<Pick<PortalProject, 'color' | 'folders' | 'icon' | 'name' | 'path'>>
): boolean {
  const entry = agentPortalProjectsFor($portalAgentProjects.get(), owner)

  if (!entry.projects.some(project => project.id === projectId)) {
    return false
  }

  updateAgentEntry(owner, {
    ...entry,
    projects: entry.projects.map(project => (project.id === projectId ? { ...project, ...patch } : project))
  })

  return true
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
  moveSessionId?: string,
  folder?: string | null
): CreatePortalProjectResult | null {
  const name = rawName.trim()

  if (!name || name.length > PORTAL_PROJECT_NAME_MAX) {
    return null
  }

  const id = `pp_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`
  const trimmedFolder = folder?.trim() || null
  const project: PortalProject = {
    createdAt: Date.now(),
    ...(trimmedFolder ? { folders: [trimmedFolder], path: trimmedFolder } : {}),
    id,
    name
  }

  return upsertPortalProject(owner, project, moveSessionId)
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
  projectId: string | null,
  options?: { detach?: boolean }
): string | undefined {
  const entry = agentPortalProjectsFor($portalAgentProjects.get(), owner)
  const rawPrevious = entry.sessionProject[sessionId]
  const previousProjectId = rawPrevious && rawPrevious !== DETACHED_PORTAL_PROJECT ? rawPrevious : undefined

  if (projectId === null) {
    if (options?.detach) {
      if (rawPrevious === DETACHED_PORTAL_PROJECT) {
        return previousProjectId
      }

      updateAgentEntry(owner, {
        ...entry,
        sessionProject: { ...entry.sessionProject, [sessionId]: DETACHED_PORTAL_PROJECT }
      })

      return previousProjectId
    }

    if (rawPrevious === undefined) {
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
