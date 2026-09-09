/** Mirrored hosted approval cards and their independent roster attention. */

import { $groupClarify, $groupHostedNeedsYou } from './group-chat'
import { groupMemberKey } from './group-membership'
import type { GroupMember, GroupPrompt } from './types'

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

export function clearHostedRoomApprovalState(group: string) {
  const prompts: Record<string, GroupPrompt> = Object.fromEntries(
    Object.entries($groupClarify.get()).filter(([, prompt]) => prompt.group !== group || !prompt.hostedApproval)
  )

  $groupClarify.set(prompts)
  const attention = { ...$groupHostedNeedsYou.get() }

  delete attention[group]
  $groupHostedNeedsYou.set(attention)
}

export function resolveHostedRoomApprovalAttention(entry: GroupPrompt) {
  const approval = entry.hostedApproval

  const remaining = Object.values($groupClarify.get()).some(prompt => {
    const pending = prompt.hostedApproval

    if (prompt.group !== entry.group || !pending) {
      return false
    }

    return !(
      prompt.requestId === entry.requestId &&
      pending.roomId === approval?.roomId &&
      pending.memberId === approval?.memberId &&
      pending.taskId === approval?.taskId &&
      pending.executionGeneration === approval?.executionGeneration
    )
  })

  $groupHostedNeedsYou.set({
    ...$groupHostedNeedsYou.get(),
    [entry.group]: remaining
  })
}

export function resetHostedRoomApprovalState() {
  $groupHostedNeedsYou.set({})
}

export function syncHostedRoomApprovals(
  group: string,
  room: { members?: unknown; room_id?: unknown },
  members: GroupMember[],
  pendingActions: unknown[]
) {
  const current = $groupClarify.get()

  const next: Record<string, GroupPrompt> = Object.fromEntries(
    Object.entries(current).filter(([, prompt]) => prompt.group !== group || !prompt.hostedApproval)
  )

  const serverMembers = Array.isArray(room.members) ? room.members : []
  let waiting = false

  for (const raw of pendingActions) {
    const action = record(raw)

    if (action?.kind !== 'approval') {
      continue
    }

    const memberId = String(action.member_id || '')
    const taskId = String(action.task_id || '')
    const requestId = String(action.request_id || '')
    const executionGeneration = Number(action.execution_generation || 0)
    const memberIndex = serverMembers.findIndex(rawMember => String(record(rawMember)?.member_id || '') === memberId)
    const member = memberIndex >= 0 ? members[memberIndex] : null
    const approval = record(action.approval)

    if (
      !member ||
      !memberId ||
      !taskId ||
      !requestId ||
      !Number.isSafeInteger(executionGeneration) ||
      executionGeneration < 1
    ) {
      continue
    }

    const key = `${group}::${groupMemberKey(member)}`
    const prior = current[key]

    const identity = {
      executionGeneration,
      memberId,
      roomId: String(room.room_id || ''),
      taskId
    }

    const choices = (Array.isArray(approval?.choices) ? approval.choices : [])
      .filter(choice => choice === 'once' || choice === 'deny')
      .map(String)

    next[key] =
      prior?.requestId === requestId &&
      prior.hostedApproval?.executionGeneration === identity.executionGeneration &&
      prior.hostedApproval.memberId === identity.memberId &&
      prior.hostedApproval.roomId === identity.roomId &&
      prior.hostedApproval.taskId === identity.taskId
        ? prior
        : {
            at: Date.now(),
            choices: choices.length ? choices : ['once', 'deny'],
            command: typeof approval?.command === 'string' ? approval.command : '',
            group,
            hostedApproval: identity,
            kind: 'approval',
            member: member.name,
            memberKey: groupMemberKey(member),
            multiSelect: false,
            question: typeof approval?.description === 'string' ? approval.description : '',
            questions: null,
            requestId,
            sessionId: null
          }
    waiting = true
  }

  $groupClarify.set(next)
  $groupHostedNeedsYou.set({
    ...$groupHostedNeedsYou.get(),
    [group]: waiting
  })
}
