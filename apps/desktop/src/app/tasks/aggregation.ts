import type { ClientSessionState } from '@/app/types'
import { sessionTitle } from '@/lib/chat-runtime'
import type { TodoItem } from '@/lib/todos'
import {
  activeSubagentCount,
  buildSubagentTree,
  failedSubagentCount,
  type SubagentNode,
  type SubagentProgress
} from '@/store/subagents'
import { todoListActive } from '@/store/todos'
import type { SessionInfo } from '@/types/hermes'

export type LiveTodoStatus = 'cancelled' | 'completed' | 'in_progress' | 'pending'

export interface LiveTaskSession {
  id: string
  routeId: null | string
  runtimeId: null | string
  session: null | SessionInfo
  title: string
  busy: boolean
  needsInput: boolean
  todos: TodoItem[]
  activeTodoCount: number
  subagents: SubagentNode[]
  activeSubagentCount: number
  failedSubagentCount: number
  updatedAt: number
}

interface AggregateLiveTaskSessionsOptions {
  sessions: SessionInfo[]
  sessionStates: Record<string, ClientSessionState>
  subagentsBySession: Record<string, SubagentProgress[]>
  todosBySession: Record<string, TodoItem[]>
}

const ACTIVE_TODO_STATUSES: ReadonlySet<LiveTodoStatus> = new Set(['pending', 'in_progress'])
const RUNTIME_KEY_PREFIX = 'runtime:'

function sessionLabel(session: null | SessionInfo, sessionId: string): string {
  return session ? sessionTitle(session) : `Session ${sessionId.slice(0, 8)}`
}

function sessionTimestampMs(session: null | SessionInfo): number {
  return session ? Math.max(session.last_active || 0, session.started_at || 0) * 1000 : 0
}

function uniqueById<T extends { id: string }>(items: readonly T[]): T[] {
  return [...new Map(items.map(item => [item.id, item])).values()]
}

export function aggregateLiveTaskSessions({
  sessions,
  sessionStates,
  subagentsBySession,
  todosBySession
}: AggregateLiveTaskSessionsOptions): LiveTaskSession[] {
  const canonicalSessionByLineage = new Map<string, SessionInfo>()

  for (const session of sessions) {
    const lineageId = session._lineage_root_id || session.id
    const current = canonicalSessionByLineage.get(lineageId)

    if (!current || sessionTimestampMs(session) >= sessionTimestampMs(current)) {
      canonicalSessionByLineage.set(lineageId, session)
    }
  }

  const sessionsByStoredId = new Map<string, SessionInfo>()

  for (const session of sessions) {
    const canonical = canonicalSessionByLineage.get(session._lineage_root_id || session.id) ?? session

    sessionsByStoredId.set(session.id, canonical)
    sessionsByStoredId.set(session._lineage_root_id || session.id, canonical)
  }

  const runtimeIdsBySession = new Map<string, Set<string>>()

  for (const runtimeId of new Set([
    ...Object.keys(sessionStates),
    ...Object.keys(todosBySession),
    ...Object.keys(subagentsBySession)
  ])) {
    const storedId = sessionStates[runtimeId]?.storedSessionId
    const canonicalStoredId = storedId ? (sessionsByStoredId.get(storedId)?.id ?? storedId) : null
    const key = canonicalStoredId ?? `${RUNTIME_KEY_PREFIX}${runtimeId}`
    const runtimeIds = runtimeIdsBySession.get(key) ?? new Set<string>()

    runtimeIds.add(runtimeId)
    runtimeIdsBySession.set(key, runtimeIds)
  }

  const rows: LiveTaskSession[] = []

  for (const [key, runtimeIdSet] of runtimeIdsBySession) {
    const runtimeOnly = key.startsWith(RUNTIME_KEY_PREFIX)
    const runtimeIds = [...runtimeIdSet]

    const runtimeId = runtimeIds.at(-1) ?? null
    const storedId = runtimeOnly ? null : key
    const session = storedId ? (sessionsByStoredId.get(storedId) ?? null) : null
    const states = runtimeIds.flatMap(id => (sessionStates[id] ? [sessionStates[id]] : []))

    const todos = uniqueById(
      runtimeIds.flatMap(id => todosBySession[id] ?? []).filter(todo => todo.id && todo.content)
    )

    const subagents = uniqueById(
      runtimeIds.flatMap(id => subagentsBySession[id] ?? []).sort((left, right) => left.updatedAt - right.updatedAt)
    )

    const activeTodos = todos.filter(todo => ACTIVE_TODO_STATUSES.has(todo.status))
    const activeSubagents = activeSubagentCount(subagents)
    const failedSubagents = failedSubagentCount(subagents)
    const busy = states.some(state => state.busy)
    const needsInput = states.some(state => state.needsInput)

    if (!(busy || needsInput || todoListActive(todos) || activeSubagents > 0)) {
      continue
    }

    rows.push({
      id: storedId ?? runtimeId ?? key,
      routeId: session?.id ?? storedId,
      runtimeId,
      session,
      title: sessionLabel(session, storedId ?? runtimeId ?? key),
      busy,
      needsInput,
      todos,
      activeTodoCount: activeTodos.length,
      subagents: buildSubagentTree(subagents),
      activeSubagentCount: activeSubagents,
      failedSubagentCount: failedSubagents,
      updatedAt: Math.max(sessionTimestampMs(session), ...subagents.map(item => item.updatedAt), 0)
    })
  }

  return rows.sort((left, right) => {
    const leftScore =
      Number(left.busy) * 4 +
      Number(left.needsInput) * 2 +
      Number(left.activeSubagentCount > 0 || left.activeTodoCount > 0)

    const rightScore =
      Number(right.busy) * 4 +
      Number(right.needsInput) * 2 +
      Number(right.activeSubagentCount > 0 || right.activeTodoCount > 0)

    return rightScore - leftScore || right.updatedAt - left.updatedAt || left.title.localeCompare(right.title)
  })
}
