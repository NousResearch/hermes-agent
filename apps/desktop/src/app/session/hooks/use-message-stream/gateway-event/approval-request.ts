import { translateNow } from '@/i18n'
import { dispatchNativeNotification } from '@/store/native-notifications'
import {
  accessTarget,
  approvalKindFields,
  type ApprovalRequest,
  networkTarget,
  networkTargetName,
  receiveApprovalRequest
} from '@/store/prompts'
import { rememberServerRequest } from '@/store/server-requests'

import { declineIfSessionStopped, type Handler, markNeedsInput, str } from './server-request-context'

/** The `approval` server request: a command, folder or site card parked for its session. */
export const approval: Handler = ctx => {
  const { request, sessionId } = ctx
  const p = request.params
  const command = str(p.command)
  const description = str(p.description) || 'dangerous command'

  if (declineIfSessionStopped(ctx)) {
    return
  }

  const parked: ApprovalRequest = {
    // false only when the backend forbids a permanent allow; it omits the field otherwise.
    allowPermanent: p.allow_permanent !== false,
    choices: Array.isArray(p.choices)
      ? p.choices.filter((choice): choice is string => typeof choice === 'string')
      : undefined,
    command,
    description,
    // The approval queue's own id — `approval.pending` / `approval.received` / `approval.respond` key on it.
    requestId: str(p.request_id) || undefined,
    serverRequestId: request.id,
    sessionId: sessionId || null,
    smartDenied: p.smart_denied === true,
    ...approvalKindFields(p)
  }

  rememberServerRequest(request)
  void receiveApprovalRequest(null, parked).catch(() => undefined)
  markNeedsInput(ctx)

  if (!request.replayed) {
    const notice = approvalNotice(parked)

    dispatchNativeNotification({
      actions: [
        {
          id: str(p.request_id) ? `approve:${str(p.request_id)}` : 'approve',
          text: translateNow(notice.approveKey)
        },
        {
          id: str(p.request_id) ? `reject:${str(p.request_id)}` : 'reject',
          text: translateNow(notice.rejectKey)
        }
      ],
      body: notice.body,
      kind: 'approval',
      namedTitleKey: notice.namedTitleKey,
      sessionId: sessionId || null,
      title: translateNow(notice.titleKey)
    })
  }
}

/** Native notification copy per approval kind: a folder, site or new-job card
 *  names what it asks for, never "Approve" a command that does not exist. */
function approvalNotice(request: ApprovalRequest) {
  const access = accessTarget(request)
  const network = networkTarget(request)

  if (request.kind === 'access' || request.kind === 'network' || request.kind === 'schedule') {
    const prefix = `${request.kind}Title`

    return {
      approveKey: 'notifications.native.allowAction',
      body: access?.path ?? (network ? networkTargetName(network) : request.description),
      namedTitleKey: `notifications.native.${prefix}Named`,
      rejectKey: 'notifications.native.denyAction',
      titleKey: `notifications.native.${prefix}`
    }
  }

  return {
    approveKey: 'notifications.native.approveAction',
    body: request.command || request.description,
    namedTitleKey: undefined,
    rejectKey: 'notifications.native.rejectAction',
    titleKey: 'notifications.native.approvalTitle'
  }
}
