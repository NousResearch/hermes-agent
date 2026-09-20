import { triggerHaptic } from '@/lib/haptics'
import { $gateway } from '@/store/gateway'
import {
  answerApproval,
  type ApprovalRequest,
  clearApprovalRequest,
  replayPendingApproval,
  sessionApprovalRequests
} from '@/store/prompts'

import type { ApprovalChoice } from './approval-choices'

/**
 * Answer one approval, whichever surface is asking.
 *
 * The order matters and is the same everywhere: tell the backend, drop the
 * prompt, then ask for whatever the queue has behind it. Answering a request
 * that is no longer pending is a no-op rather than an error — two surfaces can
 * show the same queue entry, and the second one to be pressed has nothing left
 * to say.
 */
export async function sendApproval(request: ApprovalRequest, choice: ApprovalChoice) {
  const gateway = $gateway.get()

  if (!gateway) {
    throw new Error('Gateway disconnected')
  }

  if (
    !sessionApprovalRequests(request.sessionId)
      .get()
      .some(item => item.requestId === request.requestId)
  ) {
    return
  }

  await answerApproval(gateway, request, choice)
  triggerHaptic(choice === 'deny' ? 'cancel' : 'submit')
  clearApprovalRequest(request.sessionId, request.requestId)
  void replayPendingApproval(gateway, request.sessionId).catch(() => undefined)
}
