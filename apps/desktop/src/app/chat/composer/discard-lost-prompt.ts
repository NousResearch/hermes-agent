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
