import { requestGatewayForAgent } from '@/store/gateway'
import type { SessionOwnerRoute } from '@/store/session-request-router'

export type ForeignSource = 'claude' | 'cowork' | 'codex' | 'grok'

export interface ForeignSession {
  id: string
  source: ForeignSource
  label: string
  title: string
  project?: string | null
  cwd: string | null
  mtime: number
  turn_count: number
  excerpt: string
}

export interface ForeignPage {
  sessions: ForeignSession[]
  /** Absent on older backends, which supported Claude Code and Codex only. */
  sources?: ForeignSource[]
  next_offset: number | null
  host: string
  unreadable: number
}

export interface ForeignPreview {
  truncated: boolean
  messages: { role: string; content: string }[]
  total: number
  already_imported: string | null
  cwd: string | null
}

export interface ForeignImportResult {
  session_id: string
  already_imported: boolean
}

export interface ForeignSnapshot {
  origin: { tool: string; path: string; foreign_session_id: string | null }
  messages: { role: string; content: string }[]
  title: string
}

export function foreignRequest<T>(
  owner: SessionOwnerRoute,
  method: 'list' | 'preview' | 'export' | 'import',
  params: Record<string, unknown>,
  signal?: AbortSignal
) {
  return requestGatewayForAgent<T>(
    owner.connectionId,
    owner.profile,
    `session.foreign.${method}`,
    params,
    60_000,
    signal
  )
}
