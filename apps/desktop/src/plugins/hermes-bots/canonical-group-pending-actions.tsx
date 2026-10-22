import { Button } from '@hermes/plugin-sdk'

import { type GroupMethods, offersGroupMethod } from './canonical-group-capabilities'
import type { CanonicalPendingAction } from './canonical-groups'

const ACTION_METHODS: Record<string, string> = { discard: 'groups.discard', retry: 'groups.retry', approval: 'groups.approve' }

/** One exact pending attempt, with its controls only when the gateway lists their method. */
export function CanonicalPendingActionRow({ action, methods, labels, busy, onAct, onDiscard }: {
  action: CanonicalPendingAction
  methods: GroupMethods
  labels: { allowOnce: string; deny: string; discard: string; retry: string }
  busy: boolean
  onAct: (action: CanonicalPendingAction, choice?: 'once' | 'deny') => void
  onDiscard: (action: CanonicalPendingAction) => void
}) {
  const offered = Object.hasOwn(ACTION_METHODS, action.kind) && offersGroupMethod(methods, ACTION_METHODS[action.kind])

  return <div className="flex items-center gap-2" data-task-id={action.task_id}>
    <span>{action.member_id}</span>
    {offered && action.kind === 'discard' && <Button disabled={busy} onClick={() => onDiscard({ ...action })}>{labels.discard}</Button>}
    {offered && action.kind === 'retry' && <Button disabled={busy} onClick={() => onAct({ ...action })}>{labels.retry}</Button>}
    {offered && action.kind === 'approval' && <>
      <Button disabled={busy} onClick={() => onAct({ ...action }, 'once')}>{labels.allowOnce}</Button>
      <Button disabled={busy} onClick={() => onAct({ ...action }, 'deny')}>{labels.deny}</Button>
    </>}
  </div>
}
