import { Button, ConfirmDialog } from '@hermes/plugin-sdk'
import { useRef, useState } from 'react'

import { canonicalApprovalDetails } from './canonical-group-approval'
import { CanonicalMemberFace, canonicalMemberName } from './canonical-group-identity'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { isPendingFileAction } from './canonical-groups'
import type { CanonicalPendingAction, CanonicalRoomMember } from './canonical-groups'

type Labels = ReturnType<typeof useCanonicalGroupLabels>
type Choice = 'once' | 'deny'
export interface CanonicalGroupPendingActionsProps {
  actions: CanonicalPendingAction[]
  members: CanonicalRoomMember[]
  busy?: boolean
  /** After the group moved: inherited work whose outcome is unknown says where it may have finished. */
  unknownTitle?: string
  /** Work that needs a Bot or file only the previous host has. Information only. */
  waiting?: { task_id: string; member_id: string; text: string }[]
  onAction: (action: CanonicalPendingAction, choice?: Choice) => Promise<void> | void
  onDiscard: (action: CanonicalPendingAction) => Promise<void> | void
  onRefresh?: () => Promise<void> | void
}

const attemptKey = (action: CanonicalPendingAction) => JSON.stringify([action.member_id, action.task_id, action.execution_generation])
const actionKey = (action: CanonicalPendingAction) => JSON.stringify([attemptKey(action), action.kind, action.request_id])
const text = (value: unknown) => typeof value === 'string' ? value.trim() : ''
const validAttempt = (action: CanonicalPendingAction) => Boolean(text(action.member_id) && text(action.task_id) &&
  Number.isSafeInteger(action.execution_generation) && action.execution_generation > 0)
const snapshot = (action: CanonicalPendingAction): CanonicalPendingAction => ({ ...action })

/** What the row is about. After a move, work whose outcome is unknown says where it may have finished. */
function ActionTitle({ action, labels, memberName, unknownTitle }: { action: CanonicalPendingAction; labels: Labels; memberName: string; unknownTitle?: string }) {
  const title = action.kind === 'output_retry'
    ? (!isPendingFileAction(action) ? labels.pendingFilesBlockedTitle : action.operation === 'discard' ? labels.pendingFilesCleanupTitle : labels.pendingFilesTitle)
    : action.kind === 'approval' ? labels.pendingApprovalTitle : action.kind === 'retry' ? labels.pendingRetryTitle
    : action.kind === 'stopping' ? labels.pendingStoppingTitle : unknownTitle ?? labels.pendingUnknownTitle

  if (unknownTitle && title === unknownTitle) {
    return <p className="grid min-w-0 gap-0.5 text-sm"><bdi className="font-medium">{memberName}</bdi>
      <span className="text-(--ui-text-secondary)">{unknownTitle}</span></p>
  }

  return <p className="text-sm font-medium"><bdi>{title.replace('{name}', memberName)}</bdi></p>
}

function PendingActionRow({ action, member, memberName, canSkip, busy, labels, unknownTitle, onAction, onSkip, onRefresh }: {
  action: CanonicalPendingAction; member?: CanonicalRoomMember; memberName: string; canSkip: boolean; busy: boolean; labels: Labels
  unknownTitle?: string
  onAction: CanonicalGroupPendingActionsProps['onAction']; onSkip: () => void
  onRefresh?: CanonicalGroupPendingActionsProps['onRefresh']
}) {
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const pending = useRef(false)
  const invoke = async (operation: () => Promise<void> | void) => {
    if (busy || pending.current) {return}
    pending.current = true; setSubmitting(true); setError('')
    try {await operation()} catch {setError(labels.pendingActionUnconfirmed)}
    finally {pending.current = false; setSubmitting(false)}
  }
  const disabled = busy || submitting || !validAttempt(action)
  const approval = action.kind === 'approval' ? canonicalApprovalDetails({ action, labels }) : null
  const choices = Array.isArray(action.approval?.choices) ? action.approval.choices : []
  const fileOutput = action.kind === 'output_retry'
  const filePending = isPendingFileAction(action)
  return <article aria-busy={submitting || undefined} className="grid min-w-0 gap-3 border-t border-(--ui-stroke-secondary) py-3"
    data-member-id={action.member_id} data-request-id={action.request_id} data-task-id={action.task_id} data-testid="group-chat-pending-action">
    <div className="flex items-center gap-2">
      <CanonicalMemberFace member={member} name={memberName} seed={action.member_id} />
      <ActionTitle action={action} labels={labels} memberName={memberName} unknownTitle={unknownTitle} />
    </div>
    {approval?.content}
    {fileOutput && !filePending && <p className="text-sm text-(--ui-text-secondary)" role="status">{labels.pendingFilesBlockedHelp}</p>}
    <div className="flex flex-wrap items-center justify-end gap-2">
      {approval && <>
        {choices.includes('deny') && <Button disabled={disabled || !text(action.request_id)}
          onClick={() => void invoke(() => onAction(snapshot(action), 'deny'))} size="sm" type="button" variant="secondary">{labels.deny}</Button>}
        {approval.reviewable && choices.includes('once') && <Button disabled={disabled}
          onClick={() => void invoke(() => onAction(snapshot(action), 'once'))} size="sm" type="button">{labels.allowOnce}</Button>}
        {(!approval.reviewable || !choices.some(choice => choice === 'once' || choice === 'deny')) && onRefresh && <Button disabled={busy || submitting}
          onClick={() => void invoke(onRefresh)} size="sm" type="button" variant="secondary">{labels.refresh}</Button>}
      </>}
      {fileOutput && !filePending && onRefresh && <Button disabled={busy || submitting}
        onClick={() => void invoke(onRefresh)} size="sm" type="button" variant="text">{labels.refresh}</Button>}
      {canSkip && <Button disabled={disabled} onClick={onSkip} size="sm" type="button" variant="text">{labels.skipReply}</Button>}
      {action.kind === 'retry' && <Button disabled={disabled} onClick={() => void invoke(() => onAction(snapshot(action)))} size="sm" type="button" variant="secondary">{labels.retryReply}</Button>}
    </div>
    {error && <div className="grid gap-2 text-sm text-destructive" role="alert">
      <p>{error}</p>
      {onRefresh && <div><Button disabled={busy || submitting} onClick={() => void invoke(onRefresh)} size="sm" type="button" variant="text">{labels.refresh}</Button></div>}
    </div>}
  </article>
}

/** A room's exact pending actions, presented without exposing task IDs or
 * borrowing the normal-session approval queue/response handler. */
export function CanonicalGroupPendingActions({ actions, members, busy = false, unknownTitle, waiting = [], onAction, onDiscard, onRefresh }: CanonicalGroupPendingActionsProps) {
  const labels = useCanonicalGroupLabels()
  const [discard, setDiscard] = useState<{ action: CanonicalPendingAction; name: string; notStarted: boolean } | null>(null)
  const stopping = new Set(actions.filter(action => action.kind === 'stopping').map(attemptKey))
  const retry = new Set(actions.filter(action => action.kind === 'retry').map(attemptKey))
  const rows = actions.filter(action => ['approval', 'retry', 'discard', 'stopping', 'output_retry', 'unknown'].includes(action.kind) &&
    (!stopping.has(attemptKey(action)) || action.kind === 'stopping' || action.kind === 'output_retry') && !(action.kind === 'discard' && retry.has(attemptKey(action))))
  return <div data-testid="group-chat-pending-actions">
    {rows.map(action => {
      const member = members.find(member => member.member_id === action.member_id)
      const name = canonicalMemberName(member, labels.pendingBot)
      const skip = actions.find(candidate => candidate.kind === 'discard' && attemptKey(candidate) === attemptKey(action))
      return <PendingActionRow action={action} busy={busy} canSkip={Boolean(skip) && ['retry', 'discard'].includes(action.kind)} key={actionKey(action)}
        labels={labels} member={member} memberName={name} onAction={onAction} onRefresh={onRefresh}
        onSkip={() => {if (skip) {setDiscard({ action: snapshot(skip), name, notStarted: retry.has(attemptKey(skip)) })}}} unknownTitle={unknownTitle} />
    })}
    {waiting.map(task => {
      const member = members.find(candidate => candidate.member_id === task.member_id)
      const name = canonicalMemberName(member, labels.pendingBot)

      return <article className="flex min-w-0 items-start gap-2 border-t border-(--ui-stroke-secondary) py-3" data-member-id={task.member_id}
        data-task-id={task.task_id} data-testid="group-chat-waiting-task" key={JSON.stringify([task.member_id, task.task_id])}>
        <CanonicalMemberFace member={member} name={name} seed={task.member_id} />
        <p className="grid min-w-0 gap-0.5 text-sm"><bdi className="font-medium">{name}</bdi>
          <span className="text-(--ui-text-secondary)">{task.text}</span></p>
      </article>
    })}
    <ConfirmDialog cancelLabel={labels.cancel} confirmLabel={labels.confirmDiscard} description={discard?.notStarted ? labels.skipUnstartedWarning : labels.discardWarning}
      destructive
      onClose={() => setDiscard(null)} onConfirm={async () => {
        if (!discard) {return}
        if (busy) {throw new Error(labels.pendingActionUnconfirmed)}
        try {await onDiscard(discard.action)} catch {throw new Error(labels.pendingActionUnconfirmed)}
      }} open={discard !== null} title={discard ? `${discard.name}: ${labels.discardUnknown}` : labels.discardUnknown} />
  </div>
}
