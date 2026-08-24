import type { GatewayEventPayload } from '@/lib/chat-messages'
import {
  $clarifyRequests,
  type ClarifyRequest,
  clearClarifyRequest,
  normalizeChoices,
  normalizeQuestions,
  setClarifyRequest
} from '@/store/clarify'
import type { SessionResumeResponse } from '@/types/hermes'

export interface PendingClarifyResumeState {
  authoritativeAbsent: boolean
  cleared: ClarifyRequest | null
  request: ClarifyRequest | null
}

/**
 * Restore a pending clarify from a resume/activate snapshot onto `sessionId`.
 *
 * The snapshot mirrors the live clarify.request wire shape: single-question
 * payloads carry `question`/`choices`/`multi_select`; batch (multi-question)
 * ones carry `questions` (+ any answers already locked server-side) and no
 * top-level `question`. Multi-select locks arrive as JSON-encoded arrays
 * inside a string — never a bare array — so `lockedAnswers` keeps string
 * values only.
 *
 * A missing snapshot is authoritative only for requests that already existed
 * when the RPC began. A newer clarify.request that arrives while the response
 * is in flight is left alone.
 */
export function restorePendingClarifyFromSnapshot(
  response: Pick<SessionResumeResponse, 'pending_clarify'>,
  sessionId: string,
  resumeStartedAt: number,
  requestIdAtStart?: string
): PendingClarifyResumeState {
  const pending = response.pending_clarify

  if (!pending || typeof pending.request_id !== 'string') {
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

  const questions = normalizeQuestions(pending.questions)
  const question = typeof pending.question === 'string' ? pending.question : ''

  if (!question && questions.length === 0) {
    return { authoritativeAbsent: false, cleared: null, request: null }
  }

  const choices = normalizeChoices(pending.choices)

  const lockedAnswers =
    typeof pending.answers === 'object' && pending.answers !== null
      ? Object.fromEntries(
          Object.entries(pending.answers).filter((entry): entry is [string, string] => typeof entry[1] === 'string')
        )
      : undefined

  const request: ClarifyRequest = {
    choices: choices.length > 0 ? choices : null,
    lockedAnswers,
    multiSelect: pending.multi_select === true,
    question,
    receivedAt: Date.now() / 1000,
    requestId: pending.request_id,
    sessionId,
    ...(questions.length > 0 ? { questions } : {})
  }

  setClarifyRequest(request)

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
