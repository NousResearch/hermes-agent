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

/** Restore a pending connection card from a resume snapshot with its original deadline.
 *  A missing snapshot clears only requests older than the RPC (same rule as clarify). */
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

/** Synthetic tool row for a pending operation whose `tool.start` was missed. */
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
