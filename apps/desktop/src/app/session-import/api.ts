import type { RpcMethods } from '@hermes/shared'

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

/** The three `session.foreign.*` RPCs the import dialog drives. */
export type ForeignMethod = 'session.foreign.import' | 'session.foreign.list' | 'session.foreign.preview'

export function foreignRequest<M extends ForeignMethod>(
  owner: SessionOwnerRoute,
  method: M,
  params: RpcMethods[M]['params'],
  signal?: AbortSignal
): Promise<RpcMethods[M]['result']> {
  return requestGatewayForAgent(owner.connectionId, owner.profile, method, params, 60_000, signal)
}
