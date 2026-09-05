import { atom } from 'nanostores'

import type { ProjectInfo, SessionCreateResponse } from '@/hermes'
import { requestGatewayForAgent } from '@/store/gateway'

export const codingWorkspaceDraftKey = (scope?: string | null): string => scope?.trim() || '__new__'

export interface CodingWorkspaceOwner {
  connectionId: null | string
  profile: string
  draftKey: string
}
export interface CodingWorkspaceIntent {
  projectId?: string
  path: string
  mode: 'worktree' | 'existing' | 'current' | 'folder'
  existingPath?: string
  base?: string
}
export interface CodingWorkspaceInspection {
  path: string
  repoRoot: string | null
  branch: string | null
  dirty: boolean
  branches?: string[]
  worktrees: Array<{
    path: string
    branch: string | null
    isMain?: boolean
    dirty?: boolean
    activeSessionCount?: number
  }>
}
export interface CodingWorkspacePrepared {
  requestId: string
  sourcePath: string
  cwd: string
  projectId: string
  branch: string | null
  repoRoot: string | null
}
export interface CodingWorkspaceDraft {
  owner: CodingWorkspaceOwner
  intent: CodingWorkspaceIntent | null
  controlsEnabled?: boolean
  status: 'idle' | 'inspecting' | 'ready' | 'preparing' | 'error' | 'bound'
  inspection?: CodingWorkspaceInspection
  prepared?: CodingWorkspacePrepared
  error?: string
  sessionId?: string
  createdSession?: SessionCreateResponse
  /** Original composer base retained until its pending send succeeds. */
  referenceCwd?: string | null
  requestId: string
}

export const $codingWorkspaceDrafts = atom<Record<string, CodingWorkspaceDraft>>({})
export const codingWorkspaceKey = (owner: CodingWorkspaceOwner): string =>
  JSON.stringify([owner.connectionId, owner.profile, owner.draftKey])
const preparing = new Map<string, Promise<CodingWorkspacePrepared | null>>()

function publish(key: string, draft: CodingWorkspaceDraft): void {
  $codingWorkspaceDrafts.set({ ...$codingWorkspaceDrafts.get(), [key]: draft })
}

export function resetCodingWorkspaceDraft(owner: CodingWorkspaceOwner): void {
  publish(codingWorkspaceKey(owner), {
    owner: { ...owner },
    intent: null,
    status: 'idle',
    requestId: crypto.randomUUID()
  })
}

/** One-chat presentation opt-in. Resetting the draft clears it with the intent. */
export function enableCodingWorkspaceControls(owner: CodingWorkspaceOwner): void {
  const key = codingWorkspaceKey(owner)

  if (!$codingWorkspaceDrafts.get()[key]) {resetCodingWorkspaceDraft(owner)}
  publish(key, { ...$codingWorkspaceDrafts.get()[key], controlsEnabled: true })
}

export const codingWorkspaceDraftRequestId = (owner: CodingWorkspaceOwner): string | undefined =>
  $codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)]?.requestId

export function setCodingWorkspaceIntent(owner: CodingWorkspaceOwner, intent: CodingWorkspaceIntent | null): void {
  const key = codingWorkspaceKey(owner)
  const previous = $codingWorkspaceDrafts.get()[key]

  if (previous?.sessionId || previous?.status === 'preparing' || previous?.prepared) {
    throw new Error('This workspace is already prepared. Start a new chat to change it.')
  }

  publish(key, {
    owner: { ...owner },
    intent: intent ? { ...intent } : null,
    status: 'idle',
    requestId: crypto.randomUUID()
  })
}

function current(key: string, draft: CodingWorkspaceDraft): boolean {
  return $codingWorkspaceDrafts.get()[key]?.requestId === draft.requestId
}

function request<T>(owner: CodingWorkspaceOwner, method: string, params: Record<string, unknown>): Promise<T> {
  if (owner.connectionId === '' || !owner.profile || !owner.draftKey) {
    return Promise.reject(new Error('Workspace owner is unavailable'))
  }

  return requestGatewayForAgent<T>(owner.connectionId, owner.profile, method, { ...params, profile: owner.profile })
}

export async function listCodingWorkspaceProjects(owner: CodingWorkspaceOwner): Promise<ProjectInfo[]> {
  const result = await request<{ projects: ProjectInfo[] }>(owner, 'projects.list', {})

  return result.projects
}

export async function registerCodingWorkspaceFolder(owner: CodingWorkspaceOwner, path: string): Promise<ProjectInfo> {
  const result = await request<{ project: ProjectInfo }>(owner, 'projects.workspace.register', { path })

  return result.project
}

export async function inspectCodingWorkspace(owner: CodingWorkspaceOwner): Promise<CodingWorkspaceInspection> {
  const key = codingWorkspaceKey(owner)
  const draft = $codingWorkspaceDrafts.get()[key]

  if (!draft?.intent) {throw new Error('Select a project first')}

  if (draft.prepared || draft.sessionId) {throw new Error('Workspace is already prepared')}
  publish(key, { ...draft, status: 'inspecting', error: undefined })

  try {
    const inspection = await request<CodingWorkspaceInspection>(draft.owner, 'projects.workspace.inspect', {
      path: draft.intent.path
    })

    if (!current(key, draft)) {throw new Error('Workspace draft changed during inspection')}
    const live = $codingWorkspaceDrafts.get()[key]

    if (live.status !== 'preparing' && !live.prepared) {
      publish(key, {
        ...live,
        status: 'ready',
        inspection,
        intent: inspection.repoRoot ? live.intent : { ...draft.intent, mode: 'folder' }
      })
    }

    return inspection
  } catch (error) {
    if (current(key, draft))
      {publish(key, { ...$codingWorkspaceDrafts.get()[key], status: 'error', error: String(error) })}

    throw error
  }
}

export function prepareCodingWorkspace(owner: CodingWorkspaceOwner): Promise<CodingWorkspacePrepared | null> {
  const key = codingWorkspaceKey(owner)
  const draft = $codingWorkspaceDrafts.get()[key]

  if (!draft?.intent) {return Promise.resolve(null)}

  if (draft.prepared) {return Promise.resolve(draft.prepared)}
  const operationKey = JSON.stringify([key, draft.requestId])
  const pending = preparing.get(operationKey)

  if (pending) {return pending}
  publish(key, { ...draft, status: 'preparing', error: undefined })

  const operation = request<CodingWorkspacePrepared>(draft.owner, 'projects.workspace.prepare', {
    ...draft.intent,
    requestId: draft.requestId
  })
    .then(prepared => {
      if (!current(key, draft)) {throw new Error('Workspace draft changed during preparation')}
      publish(key, { ...$codingWorkspaceDrafts.get()[key], status: 'ready', prepared })

      return prepared
    })
    .catch(error => {
      if (current(key, draft))
        {publish(key, { ...$codingWorkspaceDrafts.get()[key], status: 'error', error: String(error) })}

      throw error
    })
    .finally(() => preparing.delete(operationKey))

  preparing.set(operationKey, operation)

  return operation
}

export function rememberCodingWorkspaceSession(
  owner: CodingWorkspaceOwner,
  requestId: string | undefined,
  createdSession: SessionCreateResponse
): void {
  const key = codingWorkspaceKey(owner)
  const draft = $codingWorkspaceDrafts.get()[key]

  if (draft && draft.requestId === requestId) {publish(key, { ...draft, createdSession })}
}

export const codingWorkspaceCreatedSession = (owner: CodingWorkspaceOwner): SessionCreateResponse | undefined =>
  $codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)]?.createdSession

export function bindCodingWorkspace(owner: CodingWorkspaceOwner, sessionId: string): void {
  const key = codingWorkspaceKey(owner)
  const draft = $codingWorkspaceDrafts.get()[key]

  if (draft?.prepared) {publish(key, { ...draft, status: 'bound', sessionId })}
}

export function failCodingWorkspace(owner: CodingWorkspaceOwner, error: unknown, requestId?: string): void {
  const key = codingWorkspaceKey(owner)
  const draft = $codingWorkspaceDrafts.get()[key]

  if (draft && (!requestId || draft.requestId === requestId))
    {publish(key, { ...draft, status: 'error', error: String(error) })}
}
