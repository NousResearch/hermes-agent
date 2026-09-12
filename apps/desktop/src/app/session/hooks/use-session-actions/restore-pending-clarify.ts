import { type ChatMessage, type GatewayEventPayload, settlePendingClarifyToolCall } from '@/lib/chat-messages'
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

/** Restore-attempt observation: cold resumes learn their runtime id only on return.
 * The caller releases the listener in finally; no global tombstones survive it. */
export function observeClarifySnapshot() {
  const requestsAtStart = $clarifyRequests.get()
  const changedSessions = new Set<string>()
  let previous = requestsAtStart

  const dispose = $clarifyRequests.listen(requests => {
    for (const sessionId of new Set([...Object.keys(previous), ...Object.keys(requests)])) {
      if (previous[sessionId] !== requests[sessionId]) {changedSessions.add(sessionId)}
    }

    previous = requests
  })

  return { requestsAtStart, changedSessions, dispose }
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
  requestIdAtStart?: string,
  observation?: ReturnType<typeof observeClarifySnapshot>
): PendingClarifyResumeState {
  const pending = response.pending_clarify
  const current = $clarifyRequests.get()[sessionId]

  if (observation?.changedSessions.has(sessionId)) {
    return { authoritativeAbsent: !current, cleared: observation.requestsAtStart[sessionId] ?? null, request: current ?? null }
  }

  if (!pending || typeof pending.request_id !== 'string') {
    const existedAtStart = Boolean(current && requestIdAtStart && current.requestId === requestIdAtStart)
    const definitelyOlder = Boolean(current?.receivedAt !== undefined && current.receivedAt < resumeStartedAt)
    const legacyWithoutTime = Boolean(current && current.receivedAt === undefined && !requestIdAtStart)

    const changedIdentity = Boolean(requestIdAtStart && current && current.requestId !== requestIdAtStart)

    if (current && !changedIdentity && (existedAtStart || definitelyOlder || legacyWithoutTime)) {
      clearClarifyRequest(current.requestId, sessionId)

      return { authoritativeAbsent: true, cleared: current, request: null }
    }

    return { authoritativeAbsent: true, cleared: null, request: null }
  }

  if (current && current.requestId !== requestIdAtStart &&
      (requestIdAtStart !== undefined ||
        (current.receivedAt !== undefined && current.receivedAt >= resumeStartedAt))) {
    return { authoritativeAbsent: false, cleared: null, request: current }
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

function clarifyQuestionKey(args: unknown): string {
  if (!args || typeof args !== 'object') {return ''}
  const value = args as { question?: unknown; questions?: unknown }

  if (Array.isArray(value.questions)) {
    return JSON.stringify(value.questions.map(question => question?.question))
  }

  return typeof value.question === 'string' ? JSON.stringify([value.question]) : ''
}

/** Settle only the old request's projection, never the sole newer call.
 * Provider ids differ from request ids, so changed question text can also
 * identify the old call. Equal questions alone cannot prove it is obsolete. */
export function settleSupersededClarifyProjection(
  messages: ChatMessage[],
  previous: ClarifyRequest | null,
  current: ClarifyRequest | undefined,
  running: boolean
): ChatMessage[] {
  if (!previous || !current || previous.requestId === current.requestId) {return messages}

  const previousPayload = pendingClarifyToolPayload(previous)
  const previousQuestions = clarifyQuestionKey(previousPayload.args)
  const currentQuestions = clarifyQuestionKey(pendingClarifyToolPayload(current).args)
  let changed = false

  const next = messages.map(message => {
    const parts = message.parts.map(part => {
      if (part.type !== 'tool-call' || part.toolName !== 'clarify' ||
          part.result !== undefined || part.toolCallId === current.requestId) {return part}

      const matchesPrevious = part.toolCallId === previous.requestId ||
        (previousQuestions && previousQuestions !== currentQuestions && clarifyQuestionKey(part.args) === previousQuestions)

      if (!matchesPrevious) {return part}

      changed = true

      return settlePendingClarifyToolCall(
        [{ ...message, parts: [part] }], previousPayload, running
      ).messages[0].parts[0]
    })

    return parts.some((part, index) => part !== message.parts[index]) ? { ...message, parts } : message
  })

  return changed ? next : messages
}
