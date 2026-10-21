import { $gateway } from '@/store/gateway'
import { receiveApprovalRequest, replayPendingApproval } from '@/store/prompts'
import type { SessionResumeResult } from '@/types/hermes'

/** Re-park a resume snapshot's pending approval; true when one was restored. */
export function restorePendingApproval(response: SessionResumeResult, sessionId: string): boolean {
  const pending = response.pending_approval

  if (!pending) {
    return false
  }

  // The live `approval` server request (re-delivered from `open_requests`
  // before this ran) already parked itself with the same queue id; don't
  // clobber it with a copy that can only answer through the RPC fallback.
  void receiveApprovalRequest(null, {
    allowPermanent: pending.allow_permanent !== false,
    choices: pending.choices,
    command: pending.command ?? '',
    description: pending.description ?? 'dangerous command',
    requestId: typeof pending.request_id === 'string' ? pending.request_id : undefined,
    sessionId,
    smartDenied: pending.smart_denied === true
  })
  void replayPendingApproval($gateway.get(), sessionId).catch(() => undefined)

  return true
}
