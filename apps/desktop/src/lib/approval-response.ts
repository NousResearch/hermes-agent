import type { HermesGateway } from '@/hermes'

export type ApprovalChoice = 'once' | 'session' | 'always' | 'deny'

// Keep the RPC bounded while allowing the default backend approval window
// (approvals.timeout) to expire and return resolved: false cleanly.
export const APPROVAL_RESPONSE_TIMEOUT_MS = 90_000

export function sendApprovalResponse(
  gateway: HermesGateway,
  choice: ApprovalChoice,
  sessionId: null | string | undefined,
  requestId?: string
): Promise<{ resolved?: boolean }> {
  const payload: {
    choice: ApprovalChoice
    request_id?: string
    session_id: null | string | undefined
  } = {
    choice,
    session_id: sessionId ?? undefined
  }

  if (requestId) {
    payload.request_id = requestId
  }

  return gateway.request<{ resolved?: boolean }>(
    'approval.respond',
    payload,
    APPROVAL_RESPONSE_TIMEOUT_MS
  )
}
