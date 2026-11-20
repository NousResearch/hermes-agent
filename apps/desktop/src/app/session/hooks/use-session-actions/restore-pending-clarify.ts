import {
  type ChatMessage,
  type GatewayEventPayload,
  type PendingClarifyProjection,
  QUESTION_CARD_TOOLS,
  restorePendingClarifyToolCall
} from '@/lib/chat-messages'
import { $clarifyRequests, type ClarifyRequest, clearClarifyRequest } from '@/store/clarify'
import type { SessionResumeResult } from '@/types/hermes'

export interface PendingClarifyResumeState {
  authoritativeAbsent: boolean
  cleared: ClarifyRequest | null
  request: ClarifyRequest | null
}

/**
 * Reconcile the parked clarify for `sessionId` against a resume/activate
 * snapshot.
 *
 * The snapshot's `open_requests` names every server→client request still
 * blocking the session. The shared channel has ALREADY re-delivered those to
 * the request handlers (which parked the clarify card) before the caller sees
 * the response, so this only has to (a) report the parked request when the
 * snapshot confirms it and (b) treat a snapshot WITHOUT a clarify as
 * authoritative for requests that already existed when the RPC began — a
 * newer request that arrived while the response was in flight is left alone.
 */
export function restorePendingClarifyFromSnapshot(
  response: Pick<SessionResumeResult, 'open_requests'>,
  sessionId: string,
  resumeStartedAt: number,
  requestIdAtStart?: string
): PendingClarifyResumeState {
  const pending = (response.open_requests ?? []).find(entry => QUESTION_CARD_TOOLS.has(entry.method))

  if (!pending) {
    const current = $clarifyRequests.get()[sessionId]

    const existedAtStart = Boolean(current && requestIdAtStart && current.requestId === requestIdAtStart)
    const definitelyOlder = Boolean(current?.receivedAt !== undefined && current.receivedAt < resumeStartedAt)
    const legacyWithoutTime = Boolean(current && current.receivedAt === undefined && !requestIdAtStart)

    if (current && (existedAtStart || definitelyOlder || legacyWithoutTime)) {
      clearClarifyRequest(current.requestId, sessionId)

      return { authoritativeAbsent: true, cleared: current, request: null }
    }

    return { authoritativeAbsent: true, cleared: null, request: null }
  }

  // The request handler parked it under this session when the channel
  // re-delivered `open_requests`; a card the handler declined (empty
  // question) is simply not there.
  const parked = $clarifyRequests.get()[sessionId]

  return { authoritativeAbsent: false, cleared: null, request: parked?.requestId === pending.id ? parked : null }
}

export function pendingClarifyToolPayload(request: ClarifyRequest): GatewayEventPayload {
  return {
    args: {
      questions: request.questions.map(question => ({
        choices: question.choices ?? undefined,
        multi_select: question.multiSelect || undefined,
        question: question.question
      }))
    },
    tool_id: request.requestId
  }
}

/** Drop the pre-hydration request-id row without copying the other messages.
 *  A copied clarify row looks like a concurrent edit and hydration keeps both. */
export function withoutEarlyClarifyProjection(messages: ChatMessage[], requestId: string): ChatMessage[] {
  let changed = false
  const next: ChatMessage[] = []

  for (const message of messages) {
    const parts = message.parts.filter(
      part =>
        !(
          part.type === 'tool-call' &&
          QUESTION_CARD_TOOLS.has(part.toolName) &&
          part.result === undefined &&
          part.toolCallId === requestId
        )
    )

    if (parts.length === message.parts.length) {
      next.push(message)

      continue
    }

    changed = true

    if (parts.length > 0) {
      next.push({ ...message, parts })
    }
  }

  return changed ? next : messages
}

/**
 * The transcript a cold resume paints when the open clarify was re-delivered
 * while `session.resume` was in flight. That delivery seeds the runtime with
 * nothing but the request-id projection; treating it as the prior transcript
 * drops the stored history, and on a chat longer than one page the card then
 * remounts (losing its staged picks) when an older page lands above it.
 * Rebuild from the stored transcript and re-arm the clarify on it. Null when
 * `seeded` holds anything besides that projection, or nothing is stored.
 */
export function transcriptUnderEarlyClarify(
  seeded: ChatMessage[],
  stored: ChatMessage[],
  request: ClarifyRequest | undefined
): PendingClarifyProjection | null {
  if (!request || seeded.length === 0 || stored.length === 0) {
    return null
  }

  if (withoutEarlyClarifyProjection(seeded, request.requestId).length > 0) {
    return null
  }

  return restorePendingClarifyToolCall(stored, pendingClarifyToolPayload(request))
}
