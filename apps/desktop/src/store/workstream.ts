import { computed, type ReadableAtom } from 'nanostores'

import type { TodoItem } from '@/lib/todos'
import { $activeSessionId, $attentionSessionIds, $selectedStoredSessionId, $workingSessionIds } from '@/store/session'
import { $subagentsBySession, type SubagentStatus } from '@/store/subagents'
import { $todosBySession } from '@/store/todos'
import { $workstreamMetadata, explicitStateForLifecycle, workstreamLifecycle } from '@/store/workstream-metadata'
import type { SessionInfo } from '@/types/hermes'

export type WorkstreamState =
  | 'blocked'
  | 'close'
  | 'delegate'
  | 'done'
  | 'idle'
  | 'plan_review'
  | 'restart'
  | 'verify'
  | 'warn'
  | 'work'
  | 'workflow'

export type WorkstreamTone = 'danger' | 'done' | 'idle' | 'info' | 'muted' | 'review' | 'warn' | 'work'

export interface WorkstreamStateMeta {
  icon: string
  label: string
  tone: WorkstreamTone
}

export const WORKSTREAM_STATE_META = {
  blocked: { icon: '❗️', label: 'blocked', tone: 'danger' },
  close: { icon: '📁', label: 'safe to delete', tone: 'muted' },
  delegate: { icon: '🤖', label: 'delegating to subagents', tone: 'info' },
  done: { icon: '✅', label: 'all done', tone: 'done' },
  idle: { icon: '💤', label: 'idle', tone: 'idle' },
  plan_review: { icon: '🗳', label: 'plan review pending', tone: 'review' },
  restart: { icon: '⚡️', label: 'restart required', tone: 'warn' },
  verify: { icon: '🔎', label: 'verifying the changes', tone: 'review' },
  warn: { icon: '❓', label: 'needs your input', tone: 'warn' },
  work: { icon: '✍️', label: 'working on it now', tone: 'work' },
  workflow: { icon: '🎬', label: 'running workflow', tone: 'info' }
} satisfies Record<WorkstreamState, WorkstreamStateMeta>

type TodoLike = Partial<TodoItem>

interface SubagentLike {
  status?: SubagentStatus | string
}

export interface WorkstreamActivityInput {
  explicitState?: null | WorkstreamState
  isWorking?: boolean
  needsInput?: boolean
  session: Pick<SessionInfo, 'id'>
  subagents?: readonly SubagentLike[]
  todos?: readonly TodoLike[]
}

export interface WorkstreamActivity {
  activeSubagentCount: number
  activeTodoCount: number
  completedTodoCount: number
  failedSubagentCount: number
  icon: string
  isWorking: boolean
  label: string
  needsInput: boolean
  sessionId: string
  state: WorkstreamState
  tone: WorkstreamTone
  totalSubagentCount: number
  totalTodoCount: number
}

const ACTIVE_TODO_STATUSES = new Set<TodoItem['status']>(['in_progress', 'pending'])
const COMPLETED_TODO_STATUSES = new Set<TodoItem['status']>(['cancelled', 'completed'])
const ACTIVE_SUBAGENT_STATUSES = new Set<SubagentStatus>(['queued', 'running'])
const FAILED_SUBAGENT_STATUSES = new Set<SubagentStatus>(['failed', 'interrupted'])

function isKnownSubagentStatus(status: SubagentLike['status']): status is SubagentStatus {
  return (
    status === 'completed' ||
    status === 'failed' ||
    status === 'interrupted' ||
    status === 'queued' ||
    status === 'running'
  )
}

export function deriveWorkstreamActivity(input: WorkstreamActivityInput): WorkstreamActivity {
  const todos = input.todos ?? []
  const subagents = input.subagents ?? []
  const activeTodoCount = todos.filter(todo => todo.status && ACTIVE_TODO_STATUSES.has(todo.status)).length
  const completedTodoCount = todos.filter(todo => todo.status && COMPLETED_TODO_STATUSES.has(todo.status)).length
  const normalizedSubagentStatuses = subagents.map(subagent => subagent.status).filter(isKnownSubagentStatus)
  const activeSubagentCount = normalizedSubagentStatuses.filter(status => ACTIVE_SUBAGENT_STATUSES.has(status)).length
  const failedSubagentCount = normalizedSubagentStatuses.filter(status => FAILED_SUBAGENT_STATUSES.has(status)).length
  const isWorking = Boolean(input.isWorking)
  const needsInput = Boolean(input.needsInput)

  const state: WorkstreamState =
    input.explicitState ??
    (needsInput
      ? 'warn'
      : failedSubagentCount > 0
        ? 'blocked'
        : activeSubagentCount > 0
          ? 'delegate'
          : activeTodoCount > 0 || isWorking
            ? 'work'
            : todos.length > 0 && completedTodoCount === todos.length
              ? 'done'
              : 'idle')

  const meta = WORKSTREAM_STATE_META[state]

  return {
    activeSubagentCount,
    activeTodoCount,
    completedTodoCount,
    failedSubagentCount,
    icon: meta.icon,
    isWorking,
    label: meta.label,
    needsInput,
    sessionId: input.session.id,
    state,
    tone: meta.tone,
    totalSubagentCount: subagents.length,
    totalTodoCount: todos.length
  }
}

function sameWorkstreamActivity(a: WorkstreamActivity, b: WorkstreamActivity): boolean {
  return (
    a.activeSubagentCount === b.activeSubagentCount &&
    a.activeTodoCount === b.activeTodoCount &&
    a.completedTodoCount === b.completedTodoCount &&
    a.failedSubagentCount === b.failedSubagentCount &&
    a.icon === b.icon &&
    a.isWorking === b.isWorking &&
    a.label === b.label &&
    a.needsInput === b.needsInput &&
    a.sessionId === b.sessionId &&
    a.state === b.state &&
    a.tone === b.tone &&
    a.totalSubagentCount === b.totalSubagentCount &&
    a.totalTodoCount === b.totalTodoCount
  )
}

export function liveWorkstreamSessionId(
  sessionId: string,
  activeSessionId: null | string,
  selectedStoredSessionId: null | string
): string {
  return selectedStoredSessionId === sessionId && activeSessionId ? activeSessionId : sessionId
}

const activityAtomCache = new Map<string, ReadableAtom<WorkstreamActivity>>()

export function $workstreamActivity(sessionId: string): ReadableAtom<WorkstreamActivity> {
  const cached = activityAtomCache.get(sessionId)

  if (cached) {
    return cached
  }

  let last: WorkstreamActivity | undefined

  const runtimeAtom = computed(
    [$activeSessionId, $selectedStoredSessionId, $attentionSessionIds, $workingSessionIds, $todosBySession, $subagentsBySession],
    (activeSessionId, selectedStoredSessionId, attentionIds, workingIds, todosBySession, subagentsBySession) => {
      const liveSessionId = liveWorkstreamSessionId(sessionId, activeSessionId, selectedStoredSessionId)

      const sessionIds = liveSessionId === sessionId ? [sessionId] : [sessionId, liveSessionId]

      const todos =
        liveSessionId === sessionId
          ? todosBySession[sessionId]
          : (todosBySession[liveSessionId] ?? todosBySession[sessionId])

      const subagents =
        liveSessionId === sessionId
          ? subagentsBySession[sessionId]
          : (subagentsBySession[liveSessionId] ?? subagentsBySession[sessionId])

      return deriveWorkstreamActivity({
        isWorking: sessionIds.some(id => workingIds.includes(id)),
        needsInput: sessionIds.some(id => attentionIds.includes(id)),
        session: { id: sessionId },
        subagents,
        todos
      })
    }
  )

  const atom = computed([runtimeAtom, $workstreamMetadata], (runtimeActivity, workstreamMetadata) => {
    const explicitState = explicitStateForLifecycle(workstreamLifecycle(sessionId, workstreamMetadata))
    const next = explicitState ? { ...runtimeActivity, ...WORKSTREAM_STATE_META[explicitState], state: explicitState } : runtimeActivity

    if (last && sameWorkstreamActivity(last, next)) {
      return last
    }

    last = next

    return next
  })

  activityAtomCache.set(sessionId, atom)

  return atom
}

export function workstreamCountLabel(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`
}
