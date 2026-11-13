import type { GatewayRequest } from '@/app/session/hooks/use-prompt-actions/utils'
import { requestForOwnedSession } from '@/store/session-states'

export async function discardLostPrompt(
  sessionId: string | null | undefined,
  queueSessionKey: string,
  admissionId: string,
  request: GatewayRequest
): Promise<unknown> {
  const targetSessionId = sessionId ?? queueSessionKey

  // The owning socket's canonical protocol supplies the UNKNOWN admission's
  // generation. The current live execution generation may already be newer.
  return requestForOwnedSession(queueSessionKey, request, 'prompt.resolve_unknown', {
    session_id: targetSessionId,
    admission_id: admissionId
  })
}

/** Withdraw a server-queued admission. The authority settles every waiter and its pending fanout
 *  retires the card; a row that started meanwhile is stopped in the same RPC (no follow-up
 *  session.interrupt needed). Same owner routing as discardLostPrompt. */
export async function cancelQueuedPrompt(
  sessionId: string | null | undefined,
  queueSessionKey: string,
  admissionId: string,
  request: GatewayRequest
): Promise<unknown> {
  return requestForOwnedSession(queueSessionKey, request, 'prompt.cancel', {
    session_id: sessionId ?? queueSessionKey,
    admission_id: admissionId
  })
}

/** Deliver a server-queued admission as a steer into the running turn. The authority consumes the
 *  admission (it never also runs) and its pending fanout retires the card; a refusal
 *  (admission_started, session_idle) leaves it queued. */
export async function steerQueuedPrompt(
  sessionId: string | null | undefined,
  queueSessionKey: string,
  admissionId: string,
  request: GatewayRequest
): Promise<unknown> {
  return requestForOwnedSession(queueSessionKey, request, 'prompt.steer_queued', {
    session_id: sessionId ?? queueSessionKey,
    admission_id: admissionId
  })
}

/** Replace a server-queued admission with edited text: the authority cancels it and admits the
 *  edit as a new admission at the back of the FIFO, in one write. `submissionId` is the edit's
 *  identity: a retry of an unconfirmed save reuses it, and the owner replays a committed one. */
export async function replaceQueuedPrompt(
  sessionId: string | null | undefined,
  queueSessionKey: string,
  admissionId: string,
  text: string,
  submissionId: string,
  request: GatewayRequest
): Promise<unknown> {
  return requestForOwnedSession(queueSessionKey, request, 'prompt.replace_queued', {
    session_id: sessionId ?? queueSessionKey,
    admission_id: admissionId,
    submission_id: submissionId,
    text
  })
}
