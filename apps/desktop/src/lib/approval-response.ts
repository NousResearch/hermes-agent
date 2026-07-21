import type { HermesGateway } from '@/hermes'

export type ApprovalChoice = 'once' | 'session' | 'always' | 'deny'

export const APPROVAL_RESPONSE_TIMEOUT_MS = 310_000

export function sendApprovalResponse(
  gateway: HermesGateway,
  choice: ApprovalChoice,
  sessionId: null | string | undefined
): Promise<{ resolved?: boolean }> {
  return gateway.request<{ resolved?: boolean }>(
    'approval.respond',
    {
      choice,
      session_id: sessionId ?? undefined
    },
    APPROVAL_RESPONSE_TIMEOUT_MS
  )
}
