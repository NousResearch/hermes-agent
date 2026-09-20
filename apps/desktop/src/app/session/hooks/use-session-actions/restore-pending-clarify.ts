import type { GatewayEventPayload } from '@/lib/chat-messages'
import {
  $clarifyRequests,
  type ClarifyRequest,
  clearClarifyRequest,
  normalizeChoices,
  normalizeQuestions
} from '@/store/clarify'
import type { SessionResumeResult } from '@/types/hermes'

export interface PendingClarifyResumeState {
  authoritativeAbsent: boolean
  cleared: ClarifyRequest | null
  request: ClarifyRequest | null
}

export function pendingClarifyRequestFromSnapshot(
  response: Pick<SessionResumeResult, 'open_requests'>,
  sessionId: string
): ClarifyRequest | null {
  const pending = (response.open_requests ?? []).find(entry => entry.method === 'clarify')

  if (!pending) {
    return null
  }

  const parked = $clarifyRequests.get()[sessionId]

  if (parked?.requestId === pending.id) {
    return parked
  }

  const questions = normalizeQuestions(pending.params.questions)
  const question = typeof pending.params.question === 'string' ? pending.params.question : ''

  if (!question && questions.length === 0) {
    return null
  }

  const choices = normalizeChoices(pending.params.choices)
  const answers = pending.params.answers
  const lockedAnswers =
    typeof answers === 'object' && answers !== null
      ? Object.fromEntries(
          Object.entries(answers).filter((entry): entry is [string, string] => typeof entry[1] === 'string')
        )
      : undefined

  return {
    choices: choices.length > 0 ? choices : null,
    lockedAnswers,
    multiSelect: pending.params.multi_select === true,
    question,
    receivedAt: Date.now() / 1000,
    requestId: pending.id,
    sessionId,
    ...(questions.length > 0 ? { questions } : {})
  }
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
  requestIdAtStart?: string | null,
  preserveCurrentClarifyRequest = false
): PendingClarifyResumeState {
  if (preserveCurrentClarifyRequest) {
    return { authoritativeAbsent: false, cleared: null, request: null }
  }

  const pending = (response.open_requests ?? []).find(entry => entry.method === 'clarify')

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
  const request = parked?.requestId === pending.id ? parked : null

  return { authoritativeAbsent: false, cleared: null, request }
}

export function pendingClarifyToolPayload(request: ClarifyRequest): GatewayEventPayload {
  return {
    args: request.questions?.length
      ? {
          questions: request.questions.map(question => ({
            choices: question.choices ?? undefined,
            multi_select: question.multiSelect || undefined,
            question: question.question
          }))
        }
      : {
          choices: request.choices ?? [],
          ...(request.multiSelect ? { multi_select: true } : {}),
          question: request.question
        },
    tool_id: request.requestId
  }
}
