import type { GatewayEventPayload } from '@/lib/chat-messages'
import { parseTodoSnapshot, type TodoSnapshot } from '@/lib/todos'

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value && typeof value === 'object' && !Array.isArray(value))

function parseWithSession(value: unknown, sessionId: string): TodoSnapshot | null {
  let candidate = value

  if (typeof candidate === 'string') {
    try {
      candidate = JSON.parse(candidate)
    } catch {
      return null
    }
  }

  if (!isRecord(candidate)) {
    return null
  }

  return parseTodoSnapshot({ ...candidate, session_id: sessionId })
}

/** Read an authoritative snapshot from either a dedicated todo.updated payload
 * or a revision-stamped tool.complete payload/result. Bare lists stay display-only. */
export function todoSnapshotFromGatewayPayload(
  payload: GatewayEventPayload | undefined,
  sessionId: string
): TodoSnapshot | null {
  if (!payload || !sessionId) {
    return null
  }

  return parseWithSession(payload, sessionId) ?? parseWithSession(payload.result, sessionId)
}
