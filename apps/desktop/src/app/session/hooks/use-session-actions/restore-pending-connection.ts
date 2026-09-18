import { type ChatMessage, type GatewayEventPayload, restorePendingBlockingToolCall } from '@/lib/chat-messages'
import {
  $connectionRequests,
  clearConnectionRequest,
  type ConnectionRequest,
  normalizeConnectionRequest,
  setConnectionRequest
} from '@/store/connection-request'
import type { SessionResumeResult } from '@/types/hermes'

export interface PendingConnectionResumeState {
  authoritativeAbsent: boolean
  cleared: ConnectionRequest | null
  request: ConnectionRequest | null
}

/** Restore a pending connection card from a resume snapshot. A missing snapshot clears only
 *  requests that existed before the RPC started. */
export function restorePendingConnectionFromSnapshot(
  response: Pick<SessionResumeResult, 'pending_connection'>,
  sessionId: string,
  resumeStartedAt: number,
  opIdAtStart?: string
): PendingConnectionResumeState {
  const request = normalizeConnectionRequest(response.pending_connection, sessionId)

  if (!request) {
    const current = $connectionRequests.get()[sessionId]

    const existedAtStart = Boolean(current && opIdAtStart && current.opId === opIdAtStart)
    const definitelyOlder = Boolean(current?.receivedAt !== undefined && current.receivedAt < resumeStartedAt)

    if (current && (existedAtStart || definitelyOlder)) {
      clearConnectionRequest(current.opId, sessionId)

      return { authoritativeAbsent: true, cleared: current, request: null }
    }

    return { authoritativeAbsent: true, cleared: null, request: null }
  }

  setConnectionRequest(request)

  return { authoritativeAbsent: false, cleared: null, request }
}

/** Tool row for a pending operation whose `tool.start` event was missed. */
export function connectionRequestToolPayload(request: ConnectionRequest): GatewayEventPayload & { name: string } {
  return {
    args: {
      action: request.targets[0]?.action ?? (request.targets[0]?.kind === 'connector' ? 'connect' : 'install'),
      connectors: request.targets.map(target => ({ mcp: target.kind === 'mcp', name: target.name }))
    },
    name: 'manage_connections',
    tool_id: request.toolCallId
  }
}

/** Add the pending connection row to a projected transcript; null when there is none. */
export function projectPendingConnection(
  messages: ChatMessage[],
  request: ConnectionRequest | null
): { messages: ChatMessage[]; streamId: string } | null {
  return request ? restorePendingBlockingToolCall(messages, connectionRequestToolPayload(request)) : null
}
