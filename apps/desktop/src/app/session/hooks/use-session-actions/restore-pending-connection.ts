import type { GatewayEventPayload } from '@/lib/chat-messages'
import {
  $connectionRequests,
  clearConnectionRequest,
  type ConnectionRequest,
  normalizeConnectionRequest,
  setConnectionRequest
} from '@/store/connection-request'
import type { SessionResumeResponse } from '@/types/hermes'

export interface PendingConnectionResumeState {
  authoritativeAbsent: boolean
  cleared: ConnectionRequest | null
  request: ConnectionRequest | null
}

/**
 * Restore a pending connection operation from a resume/activate snapshot onto
 * `sessionId`. The snapshot is the live `connection.request` wire payload, so
 * the restored card carries the SAME `deadline_at` the backend fixed when the
 * tool call began — reopening the chat never buys more time.
 *
 * A missing snapshot is authoritative only for requests that already existed
 * when the RPC began; a newer request that arrives while the response is in
 * flight is left alone (same rule as `restorePendingClarifyFromSnapshot`).
 */
export function restorePendingConnectionFromSnapshot(
  response: Pick<SessionResumeResponse, 'pending_connection'>,
  sessionId: string,
  resumeStartedAt: number,
  requestIdAtStart?: string
): PendingConnectionResumeState {
  const request = normalizeConnectionRequest(response.pending_connection, sessionId)

  if (!request) {
    const current = $connectionRequests.get()[sessionId]

    const existedAtStart = Boolean(current && requestIdAtStart && current.requestId === requestIdAtStart)
    const definitelyOlder = Boolean(current?.receivedAt !== undefined && current.receivedAt < resumeStartedAt)

    if (current && (existedAtStart || definitelyOlder)) {
      clearConnectionRequest(current.requestId, sessionId)

      return { authoritativeAbsent: true, cleared: current, request: null }
    }

    return { authoritativeAbsent: true, cleared: null, request: null }
  }

  setConnectionRequest(request)

  return { authoritativeAbsent: false, cleared: null, request }
}

/** The tool row a pending operation renders as when `tool.start` was never
 *  seen (reconnect / resume): the folded tool call, MCP targets, request id
 *  as the stable row id. */
export function connectionRequestToolPayload(request: ConnectionRequest): GatewayEventPayload {
  return {
    args: {
      action: request.targets[0]?.action ?? 'install',
      connectors: request.targets.map(target => ({ mcp: target.kind === 'mcp', name: target.name })),
      reason: request.reason
    },
    name: 'manage_connections',
    tool_id: request.requestId
  }
}
