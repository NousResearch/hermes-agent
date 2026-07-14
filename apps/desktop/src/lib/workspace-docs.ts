// Typed client for the workspace-documents REST endpoints
// (`hermes_cli/web_server.py`'s `/api/workspace-docs/*`). Documents are inert
// safekeeping Markdown files under `<workspaceRoot>/.hermes/docs`; this client
// only does list/read/write/archive transport — no MDX rendering, import,
// export, or promotion. Always goes through `window.hermesDesktop.api` (there
// is no local Electron IPC counterpart), routed to the active profile like
// `desktop-fs.ts` does so a remote profile never reads the wrong backend.

import { desktopFsProfile } from './desktop-fs'

export type WorkspaceDocType =
  | 'skill-template'
  | 'memory-note'
  | 'workspace-instructions'
  | 'prompt-template'
  | 'runbook'
  | 'generic-md'

export type WorkspaceDocStatus = 'draft' | 'ready' | 'archived'

export type WorkspaceDocApplyState = 'unapplied' | 'exported' | 'imported'

export interface WorkspaceDocFrontmatterJson {
  docType: WorkspaceDocType
  title: string
  workspaceId?: string | null
  createdAt?: string | null
  updatedAt?: string | null
  status: WorkspaceDocStatus
  applyState: WorkspaceDocApplyState
  description?: string | null
  tags: string[]
}

export interface WorkspaceDocSummary extends Partial<WorkspaceDocFrontmatterJson> {
  path: string
  valid: boolean
  error?: string
}

export interface WorkspaceDocDetail {
  path: string
  content: string
  body: string
  frontmatter: WorkspaceDocFrontmatterJson
}

export interface WorkspaceDocWriteFrontmatter {
  docType: WorkspaceDocType
  title: string
  status?: WorkspaceDocStatus
  applyState?: WorkspaceDocApplyState
  description?: string | null
  tags?: string[]
  workspaceId?: string | null
  createdAt?: string | null
}

export interface WorkspaceDocWriteResult {
  ok: boolean
  path: string
  frontmatter: WorkspaceDocFrontmatterJson
}

function bridge() {
  const desktop = window.hermesDesktop

  if (!desktop) {
    throw new Error('Hermes Desktop bridge is unavailable')
  }

  return desktop
}

function docsApi<T>(path: string, body?: Record<string, unknown>): Promise<T> {
  return bridge().api<T>(
    body
      ? { body, method: 'POST', path, profile: desktopFsProfile() }
      : { path, profile: desktopFsProfile() }
  )
}

function docsQuery(params: Record<string, string>): string {
  return Object.entries(params)
    .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
    .join('&')
}

function frontmatterToPayload(frontmatter: WorkspaceDocWriteFrontmatter): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    doc_type: frontmatter.docType,
    title: frontmatter.title
  }

  if (frontmatter.status !== undefined) {
    payload.status = frontmatter.status
  }
  if (frontmatter.applyState !== undefined) {
    payload.apply_state = frontmatter.applyState
  }
  if (frontmatter.description !== undefined) {
    payload.description = frontmatter.description
  }
  if (frontmatter.tags !== undefined) {
    payload.tags = frontmatter.tags
  }
  if (frontmatter.workspaceId !== undefined) {
    payload.workspace_id = frontmatter.workspaceId
  }
  if (frontmatter.createdAt !== undefined) {
    payload.created_at = frontmatter.createdAt
  }

  return payload
}

export async function listWorkspaceDocs(workspaceRoot: string): Promise<WorkspaceDocSummary[]> {
  const result = await docsApi<{ documents: WorkspaceDocSummary[] }>(
    `/api/workspace-docs/list?${docsQuery({ workspaceRoot })}`
  )

  return result.documents ?? []
}

export async function readWorkspaceDoc(workspaceRoot: string, path: string): Promise<WorkspaceDocDetail> {
  return docsApi<WorkspaceDocDetail>(`/api/workspace-docs/read?${docsQuery({ path, workspaceRoot })}`)
}

export async function writeWorkspaceDoc(
  workspaceRoot: string,
  path: string,
  frontmatter: WorkspaceDocWriteFrontmatter,
  body = ''
): Promise<WorkspaceDocWriteResult> {
  return docsApi<WorkspaceDocWriteResult>('/api/workspace-docs', {
    body,
    frontmatter: frontmatterToPayload(frontmatter),
    path,
    workspaceRoot
  })
}

export async function archiveWorkspaceDoc(workspaceRoot: string, path: string): Promise<WorkspaceDocWriteResult> {
  return docsApi<WorkspaceDocWriteResult>('/api/workspace-docs/archive', { path, workspaceRoot })
}
