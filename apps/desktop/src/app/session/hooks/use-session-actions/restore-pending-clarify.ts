import { type GatewayEventPayload, QUESTION_CARD_TOOLS } from '@/lib/chat-messages'
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
  if (request.setup) {
    return {
      args: {
        intent: request.setup.intent,
        kind: request.setup.kind,
        multi_select: request.setup.multiSelect,
        options: request.setup.options ?? undefined,
        question: request.questions[0]?.question
      },
      name: 'setup_choose',
      tool_id: request.requestId
    }
  }

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
