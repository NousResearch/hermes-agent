import { agentSidebarKey, type AgentSidebarOwner } from '@/store/agent-sidebar-preferences'
import { activeGatewayConnectionId, requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import {
  type BackendProjectRecord,
  createPortalProject,
  type CreatePortalProjectResult,
  deletePortalProject,
  mergeBackendPortalProjects,
  moveSessionToPortalProject,
  patchPortalProject,
  PORTAL_PROJECT_NAME_MAX,
  type PortalProject,
  projectInfoToPortalProject,
  renamePortalProject,
  upsertPortalProject
} from '@/store/portal-projects'
import { $activeGatewayProfile } from '@/store/profile'
import { refreshProjects, refreshProjectTree } from '@/store/projects'
import { $sessions, sessionMatchesStoredId } from '@/store/session'

const SYNC_TIMEOUT_MS = 8_000
const inFlightSync = new Map<string, Promise<void>>()

function isActiveOwner(owner: AgentSidebarOwner): boolean {
  const activeConn = activeGatewayConnectionId()
  const activeProfile = ($activeGatewayProfile.get() || 'default').trim().toLowerCase()
  const ownerProfile = (owner.profile || 'default').trim().toLowerCase()

  return (owner.connectionId ?? null) === (activeConn ?? null) && ownerProfile === activeProfile
}

function refreshActiveProjectStores(owner: AgentSidebarOwner): void {
  if (!isActiveOwner(owner)) {
    return
  }

  void refreshProjects()
  void refreshProjectTree()
}

function isOfflineOrUnsupportedRpc(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err ?? '')

  return /not open|not connected|timed out|connection closed|method not found|unknown method|-32601/i.test(msg)
}

/** Owner-routed `session.workspace.move`: updates the backend session's workspace `cwd` directly
 *  on the owning machine/profile without depending on the active gateway's `$projectTree`. */
async function moveWorkspaceSession(owner: AgentSidebarOwner, sessionId: string, cwd: string): Promise<void> {
  await requestGatewayForAgent(
    owner.connectionId,
    owner.profile,
    'session.workspace.move',
    { cwd, profile: owner.profile, session_key: sessionId },
    SYNC_TIMEOUT_MS,
    undefined,
    { spawnPriority: 'foreground' }
  )

  $sessions.set(
    $sessions.get().map(row =>
      sessionMatchesStoredId(row, sessionId) &&
      (row.connection_id ?? null) === (owner.connectionId ?? null) &&
      (row.profile || 'default').toLowerCase() === (owner.profile || 'default').toLowerCase()
        ? { ...row, cwd }
        : row
    )
  )
  refreshActiveProjectStores(owner)
}
/** Pull backend `projects.db` projects for an agent and merge them into `$portalAgentProjects`.
 *  Deduplicates concurrent requests per agent and degrades quietly when offline or unsupported. */
export function syncPortalAgentProjects(owner: AgentSidebarOwner): Promise<void> {
  const key = agentSidebarKey(owner)
  const existing = inFlightSync.get(key)

  if (existing) {
    return existing
  }

  const job = requestGatewayForAgent<{ projects?: BackendProjectRecord[] }>(
    owner.connectionId,
    owner.profile,
    'projects.list',
    { profile: owner.profile },
    SYNC_TIMEOUT_MS
  )
    .then(payload => {
      mergeBackendPortalProjects(owner, payload?.projects ?? [])
    })
    .catch(() => {
      // Offline or older backend without projects.list: retain cached / local projects untouched.
    })
    .finally(() => {
      inFlightSync.delete(key)
    })

  inFlightSync.set(key, job)

  return job
}

export interface CreateSyncedProjectOptions {
  folder?: string | null
  moveSessionId?: string
}

/** Create a project backed by `projects.db` (supports both folder-less and folder-backed projects).
 *  Falls back to local persistence if the backend is unavailable. */
export async function createPortalProjectSynced(
  owner: AgentSidebarOwner,
  rawName: string,
  options: CreateSyncedProjectOptions = {}
): Promise<CreatePortalProjectResult | null> {
  const name = rawName.trim()

  if (!name || name.length > PORTAL_PROJECT_NAME_MAX) {
    return null
  }

  const folder = options.folder?.trim() || null

  try {
    const res = await requestGatewayForAgent<{ project?: BackendProjectRecord | null }>(
      owner.connectionId,
      owner.profile,
      'projects.create',
      {
        folders: folder ? [folder] : [],
        name,
        ...(folder ? { primary_path: folder } : {}),
        profile: owner.profile
      },
      SYNC_TIMEOUT_MS,
      undefined,
      { spawnPriority: 'foreground' }
    )

    if (res?.project) {
      const mapped = projectInfoToPortalProject(res.project)
      const result = upsertPortalProject(owner, mapped, options.moveSessionId)

      if (options.moveSessionId && mapped.path) {
        void moveWorkspaceSession(owner, options.moveSessionId, mapped.path).catch(err =>
          notifyError(err, 'Could not move chat folder')
        )
      }

      refreshActiveProjectStores(owner)

      return result
    }
  } catch (err) {
    if (!isOfflineOrUnsupportedRpc(err)) {
      notifyError(err, 'Could not create project')

      return null
    }
  }

  return createPortalProject(owner, name, options.moveSessionId, folder)
}

export function deletePortalProjectSynced(
  owner: AgentSidebarOwner,
  project: Pick<PortalProject, 'backend' | 'id'>,
  restoreMove?: { sessionId: string; previousProjectId?: string; previousProjectPath?: string | null }
): void {
  deletePortalProject(owner, project.id, restoreMove)

  if (restoreMove?.previousProjectPath) {
    void moveWorkspaceSession(owner, restoreMove.sessionId, restoreMove.previousProjectPath).catch(() => {})
  }

  if (!project.backend) {
    return
  }

  void requestGatewayForAgent(
    owner.connectionId,
    owner.profile,
    'projects.delete',
    { id: project.id, profile: owner.profile },
    SYNC_TIMEOUT_MS,
    undefined,
    { spawnPriority: 'foreground' }
  )
    .then(() => refreshActiveProjectStores(owner))
    .catch(err => {
      notifyError(err, 'Could not delete project')
      void syncPortalAgentProjects(owner)
    })
}

export function renamePortalProjectSynced(
  owner: AgentSidebarOwner,
  project: Pick<PortalProject, 'backend' | 'id'>,
  rawName: string
): boolean {
  const ok = renamePortalProject(owner, project.id, rawName)

  if (!ok || !project.backend) {
    return ok
  }

  void requestGatewayForAgent(
    owner.connectionId,
    owner.profile,
    'projects.update',
    { id: project.id, name: rawName.trim(), profile: owner.profile },
    SYNC_TIMEOUT_MS,
    undefined,
    { spawnPriority: 'foreground' }
  )
    .then(() => refreshActiveProjectStores(owner))
    .catch(err => {
      notifyError(err, 'Could not rename project')
      void syncPortalAgentProjects(owner)
    })

  return true
}

export function setPortalProjectAppearanceSynced(
  owner: AgentSidebarOwner,
  project: Pick<PortalProject, 'backend' | 'id'>,
  patch: { color?: null | string; icon?: null | string }
): void {
  patchPortalProject(owner, project.id, patch)

  if (!project.backend) {
    return
  }

  void requestGatewayForAgent(
    owner.connectionId,
    owner.profile,
    'projects.update',
    {
      id: project.id,
      profile: owner.profile,
      ...(patch.color !== undefined ? { color: patch.color ?? '' } : {}),
      ...(patch.icon !== undefined ? { icon: patch.icon ?? '' } : {})
    },
    SYNC_TIMEOUT_MS,
    undefined,
    { spawnPriority: 'foreground' }
  )
    .then(() => refreshActiveProjectStores(owner))
    .catch(err => {
      notifyError(err, 'Could not update project appearance')
      void syncPortalAgentProjects(owner)
    })
}

export function addPortalProjectFolderSynced(
  owner: AgentSidebarOwner,
  project: Pick<PortalProject, 'backend' | 'folders' | 'id' | 'path'>,
  folderPath: string
): void {
  const trimmed = folderPath.trim()

  if (!trimmed) {
    return
  }

  const existing = project.folders?.length ? [...project.folders] : project.path ? [project.path] : []

  if (!existing.includes(trimmed)) {
    existing.push(trimmed)
  }

  const nextPath = project.path || trimmed
  patchPortalProject(owner, project.id, { folders: existing, path: nextPath })

  if (!project.backend) {
    return
  }

  void requestGatewayForAgent(
    owner.connectionId,
    owner.profile,
    'projects.add_folder',
    { id: project.id, is_primary: !project.path, path: trimmed, profile: owner.profile },
    SYNC_TIMEOUT_MS,
    undefined,
    { spawnPriority: 'foreground' }
  )
    .then(() => refreshActiveProjectStores(owner))
    .catch(err => {
      notifyError(err, 'Could not attach folder')
      void syncPortalAgentProjects(owner)
    })
}

/** Move a session to a portal project (or detach back to loose chats when `project` is null).
 *  When moving into a backend folder-backed project, also updates the session's workspace cwd. */
export function moveSessionToPortalProjectSynced(
  owner: AgentSidebarOwner,
  sessionId: string,
  project: Pick<PortalProject, 'backend' | 'id' | 'path'> | null,
  options?: { detach?: boolean }
): string | undefined {
  const previous = moveSessionToPortalProject(owner, sessionId, project?.id ?? null, options)

  if (project?.backend && project.path) {
    void moveWorkspaceSession(owner, sessionId, project.path).catch(err =>
      notifyError(err, 'Could not move chat folder')
    )
  }

  return previous
}
