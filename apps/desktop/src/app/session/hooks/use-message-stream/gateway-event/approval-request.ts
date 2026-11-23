import { translateNow } from '@/i18n'
import { dispatchNativeNotification } from '@/store/native-notifications'
import {
  accessTarget,
  type ApprovalAccessTarget,
  approvalKindFields,
  approvalNamesNothing,
  type ApprovalRequest,
  networkTarget,
  networkTargetName,
  receiveApprovalRequest
} from '@/store/prompts'
import { rememberServerRequest } from '@/store/server-requests'

import { declineIfSessionStopped, type Handler, markNeedsInput, str } from './server-request-context'

/** JSON-RPC "invalid params". */
const JSON_RPC_INVALID_PARAMS = -32602

/** The `approval` server request: a command, folder or site card parked for its session. */
export const approval: Handler = ctx => {
  const { request, sessionId } = ctx
  const p = request.params
  const command = str(p.command)
  const description = str(p.description) || 'dangerous command'

  if (declineIfSessionStopped(ctx)) {
    return
  }

  const kindFields = approvalKindFields(p)

  // A folder or site request that names no folder or site is refused: its sender gets an error
  // and no card is shown.
  if (approvalNamesNothing(kindFields)) {
    request.fail(JSON_RPC_INVALID_PARAMS, 'an access request needs a path and a network request needs a host')

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
    ...kindFields
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

/** Native notification copy per approval kind: a folder or site card names
 *  what it asks for, never "Approve" a command that does not exist. */
function approvalNotice(request: ApprovalRequest) {
  const access = accessTarget(request)
  const network = networkTarget(request)

  if (request.kind === 'access' || request.kind === 'network') {
    const prefix = request.kind === 'access' ? 'accessTitle' : 'networkTitle'

    return {
      approveKey: 'notifications.native.allowAction',
      body: access ? accessNoticeBody(access) : network ? networkTargetName(network) : request.description,
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

/** A folder card's notification text: the folder, led by whether the request is to read it or to
 *  read and write it. The notification's Allow button answers the card. */
function accessNoticeBody(target: ApprovalAccessTarget): string {
  if (target.access === 'read_write') {
    return translateNow('notifications.native.accessReadWriteBody', target.path)
  }

  return target.access === 'read' ? translateNow('notifications.native.accessReadBody', target.path) : target.path
}
