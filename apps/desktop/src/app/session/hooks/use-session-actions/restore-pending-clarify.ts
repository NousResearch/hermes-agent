import type { GatewayEventPayload } from '@/lib/chat-messages'
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
/**
 * Backwards compat: a v0.21.2 backend predates `open_requests` and reports a
 * still-pending clarify on resume as `pending_clarify` (old shape). Synthesize
 * the `open_requests` entry the reconciler below already understands so a resume
 * from an old backend still restores the parked card instead of clearing it
 * ("Result unavailable" on the sealed transcript row).
 */
function legacyClarifyOpenRequests(
  pending: SessionResumeResult['pending_clarify']
): NonNullable<SessionResumeResult['open_requests']> {
  if (!pending || typeof pending.request_id !== 'string' || !pending.request_id) {
    return []
  }

  return [{ id: pending.request_id, method: 'clarify', params: pending }]
}

export function restorePendingClarifyFromSnapshot(
  response: Pick<SessionResumeResult, 'open_requests' | 'pending_clarify'>,
  sessionId: string,
  resumeStartedAt: number,
  requestIdAtStart?: string
): PendingClarifyResumeState {
  const openRequests = response.open_requests ?? legacyClarifyOpenRequests(response.pending_clarify)
  const pending = openRequests.find(entry => entry.method === 'clarify')

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
