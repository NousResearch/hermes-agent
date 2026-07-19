import { describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import type { TodoItem } from '@/lib/todos'
import type { SubagentProgress, SubagentStatus } from '@/store/subagents'
import type { SessionInfo } from '@/types/hermes'

import { aggregateLiveTaskSessions } from './aggregation'

const session = (id: string, overrides: Partial<SessionInfo> = {}): SessionInfo =>
  ({
    archived: false,
    cwd: '/tmp/project',
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: true,
    last_active: 10,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 1,
    title: 'Mapped session',
    tool_call_count: 0,
    ...overrides
  }) as SessionInfo

const pendingTodo = (id: string): TodoItem => ({ content: `task ${id}`, id, status: 'pending' })
const completedTodo = (id: string): TodoItem => ({ content: `task ${id}`, id, status: 'completed' })

const subagent = (id: string, status: SubagentStatus, updatedAt: number): SubagentProgress => ({
  filesRead: [],
  filesWritten: [],
  goal: `subagent ${id}`,
  id,
  parentId: null,
  startedAt: 1,
  status,
  stream: [],
  taskCount: 1,
  taskIndex: 0,
  updatedAt
})

describe('aggregateLiveTaskSessions', () => {
  it('normalizes a mapped runtime id to its stored session', () => {
    const runtimeState = createClientSessionState('stored-1')

    const rows = aggregateLiveTaskSessions({
      sessions: [session('stored-1')],
      sessionStates: { 'runtime-1': runtimeState },
      subagentsBySession: {},
      todosBySession: { 'runtime-1': [pendingTodo('mapped')] }
    })

    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({
      id: 'stored-1',
      routeId: 'stored-1',
      runtimeId: 'runtime-1',
      title: 'Mapped session'
    })
    expect(rows[0]?.todos.map(todo => todo.id)).toEqual(['mapped'])
  })

  it('keeps an unmapped runtime id visible without inventing a stored route', () => {
    const rows = aggregateLiveTaskSessions({
      sessions: [],
      sessionStates: {},
      subagentsBySession: {},
      todosBySession: { 'runtime-only': [pendingTodo('unmapped')] }
    })

    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({
      id: 'runtime-only',
      routeId: null,
      runtimeId: 'runtime-only',
      title: 'Session runtime-'
    })
  })

  it('keeps unmapped runtime state visible with its live status', () => {
    const runtimeState = createClientSessionState(null)
    runtimeState.busy = true
    runtimeState.needsInput = true

    const rows = aggregateLiveTaskSessions({
      sessions: [],
      sessionStates: { 'runtime-only': runtimeState },
      subagentsBySession: {},
      todosBySession: {}
    })

    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({
      busy: true,
      id: 'runtime-only',
      needsInput: true,
      routeId: null,
      runtimeId: 'runtime-only'
    })
  })

  it('combines live data from multiple runtimes for one stored session', () => {
    const first = createClientSessionState('stored-1')
    first.busy = true
    const second = createClientSessionState('stored-1')
    second.needsInput = true

    const rows = aggregateLiveTaskSessions({
      sessions: [session('stored-1')],
      sessionStates: { 'runtime-1': first, 'runtime-2': second },
      subagentsBySession: {},
      todosBySession: { 'runtime-1': [pendingTodo('first')], 'runtime-2': [pendingTodo('second')] }
    })

    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({
      busy: true,
      id: 'stored-1',
      needsInput: true,
      routeId: 'stored-1'
    })
    expect(rows[0]?.todos.map(todo => todo.id)).toEqual(['first', 'second'])
  })

  it('deduplicates logical work across overlapping runtimes before counting', () => {
    const oldRuntime = createClientSessionState('stored-1')
    const newRuntime = createClientSessionState('stored-1')

    const rows = aggregateLiveTaskSessions({
      sessions: [session('stored-1')],
      sessionStates: { old: oldRuntime, current: newRuntime },
      subagentsBySession: {
        old: [subagent('agent-1', 'running', 1)],
        current: [subagent('agent-1', 'completed', 2)]
      },
      todosBySession: {
        old: [pendingTodo('todo-1')],
        current: [completedTodo('todo-1'), pendingTodo('todo-2')]
      }
    })

    expect(rows).toHaveLength(1)
    expect(rows[0]?.todos).toEqual([completedTodo('todo-1'), pendingTodo('todo-2')])
    expect(rows[0]).toMatchObject({ activeSubagentCount: 0, activeTodoCount: 1, failedSubagentCount: 0 })
  })

  it('canonicalizes lineage roots and continuations into one conversation row', () => {
    const rootRuntime = createClientSessionState('root')
    const continuationRuntime = createClientSessionState('continuation')

    const rows = aggregateLiveTaskSessions({
      sessions: [session('root'), session('continuation', { _lineage_root_id: 'root', title: 'Continuation' })],
      sessionStates: { rootRuntime, continuationRuntime },
      subagentsBySession: {},
      todosBySession: {
        rootRuntime: [pendingTodo('root-task')],
        continuationRuntime: [pendingTodo('continuation-task')]
      }
    })

    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({ id: 'continuation', routeId: 'continuation', title: 'Continuation' })
    expect(rows[0]?.todos.map(todo => todo.id)).toEqual(['root-task', 'continuation-task'])
  })
})
