import { Button } from '@hermes/plugin-sdk'

import { memberName } from './canonical-group-activity'
import { canonicalApprovalPreview } from './canonical-group-approval'
import { type GroupMethods, offersGroupMethod } from './canonical-group-capabilities'
import type { CanonicalPendingAction, CanonicalRoomMember } from './canonical-groups'

const ACTION_METHODS: Record<string, string> = { discard: 'groups.discard', retry: 'groups.retry', approval: 'groups.approve' }

interface ApprovalLabels {
  approvalAction: string
  approvalCommand: string
  approvalDetailsMissing: string
}

function CanonicalApprovalDetails({ action, labels }: { action: CanonicalPendingAction; labels: ApprovalLabels }) {
  const { actualCommand, description, reviewable } = canonicalApprovalPreview(action)

  return <div className="grid min-w-0 gap-2 text-xs">
    {description && description !== actualCommand && <div>
      <p className="text-(--ui-text-secondary)">{labels.approvalAction}</p>
      <p className="whitespace-pre-wrap break-words">{description}</p>
    </div>}
    {actualCommand && <div>
      <p className="text-(--ui-text-secondary)">{labels.approvalCommand}</p>
      <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-xs">{actualCommand}</pre>
    </div>}
    {!reviewable && <p className="text-(--ui-text-secondary)">{labels.approvalDetailsMissing}</p>}
  </div>
}

interface PendingActionLabels extends ApprovalLabels {
  pendingApprovalTitle: string
  pendingRetryTitle: string
  pendingUnknownTitle: string
  allowOnce: string
  deny: string
  discard: string
  refresh: string
  retry: string
}

/** One exact pending attempt, named by its Bot, with only the choices the gateway offered and only
 *  the controls whose method it lists. */
export function CanonicalPendingActionRow({ action, members = [], methods, labels, busy, onAct, onDiscard, onRefresh }: {
  action: CanonicalPendingAction
  members?: CanonicalRoomMember[]
  methods: GroupMethods
  labels: PendingActionLabels
  busy: boolean
  onAct: (action: CanonicalPendingAction, choice?: 'once' | 'deny') => void
  onDiscard: (action: CanonicalPendingAction) => void
  onRefresh: () => void
}) {
  const member = members.find(candidate => candidate.member_id === action.member_id)
  const name = member ? memberName(member) : action.member_id
  const approval = action.kind === 'approval'
  const offered = Object.hasOwn(ACTION_METHODS, action.kind) && offersGroupMethod(methods, ACTION_METHODS[action.kind])
  const choices = offered && Array.isArray(action.approval?.choices) ? action.approval.choices : []
  const reviewable = approval && canonicalApprovalPreview(action).reviewable

  const title = approval ? labels.pendingApprovalTitle
    : action.kind === 'retry' ? labels.pendingRetryTitle : labels.pendingUnknownTitle

  return <div className="grid gap-2" data-member-id={action.member_id} data-task-id={action.task_id}>
    <p className="text-xs font-medium">{title.replace('{name}', name)}</p>
    {approval && <CanonicalApprovalDetails action={action} labels={labels} />}
    <div className="flex flex-wrap items-center gap-2">
      {offered && action.kind === 'discard' && <Button disabled={busy} onClick={() => onDiscard({ ...action })}>{labels.discard}</Button>}
      {offered && action.kind === 'retry' && <Button disabled={busy} onClick={() => onAct({ ...action })}>{labels.retry}</Button>}
      {reviewable && choices.includes('once') && <Button disabled={busy} onClick={() => onAct({ ...action }, 'once')}>{labels.allowOnce}</Button>}
      {approval && choices.includes('deny') && <Button disabled={busy} onClick={() => onAct({ ...action }, 'deny')}>{labels.deny}</Button>}
      {approval && !reviewable && <Button disabled={busy} onClick={onRefresh}>{labels.refresh}</Button>}
    </div>
  </div>
}
