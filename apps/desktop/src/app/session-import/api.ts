import { requestGatewayForAgent } from '@/store/gateway'
import type { SessionOwnerRoute } from '@/store/session-request-router'

export interface ForeignSession {
  id: string
  source: 'claude' | 'codex'
  label: string
  title: string
  cwd: string | null
  mtime: number
  turn_count: number
  excerpt: string
}

export interface ForeignPage {
  sessions: ForeignSession[]
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

export function foreignRequest<T>(
  owner: SessionOwnerRoute,
  method: 'list' | 'preview' | 'import',
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
