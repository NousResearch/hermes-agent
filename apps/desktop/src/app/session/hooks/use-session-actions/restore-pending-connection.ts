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

  const current = $connectionRequests.get()[sessionId]

  // A resume snapshot is read once and can land after the live frames it predates. It must not
  // revive a card the operation already settled, nor put back a row a newer frame has moved.
  // Only the same operation can refuse it: a settled cache says nothing about the next operation
  // the session opened. A settled refusal is no pending card; a newer live frame is still the
  // pending card, and the caller must keep treating the session as waiting on it.
  if (current?.opId === request.opId && (current.settled || current.seq > request.seq)) {
    return { authoritativeAbsent: false, cleared: null, request: current.settled ? null : current }
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
