import { beforeEach, describe, expect, it } from 'vitest'

import type { TodoItem } from '@/lib/todos'
import { $activeSessionId, $attentionSessionIds, $selectedStoredSessionId, $workingSessionIds } from '@/store/session'
import { $subagentsBySession, type SubagentProgress } from '@/store/subagents'
import { $todosBySession } from '@/store/todos'
import { $workstreamMetadata } from '@/store/workstream-metadata'
import type { SessionInfo } from '@/types/hermes'

import {
  $workstreamActivity,
  deriveWorkstreamActivity,
  WORKSTREAM_STATE_META,
  type WorkstreamState
} from './workstream'

const WORKSTREAM_STATES: WorkstreamState[] = [
  'work',
  'verify',
  'done',
  'close',
  'blocked',
  'warn',
  'delegate',
  'workflow',
  'plan_review',
  'restart',
  'idle'
]

const session = (over: Partial<SessionInfo> = {}): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id: 's1',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: null,
  tool_call_count: 0,
  ...over
})

const todo = (status: TodoItem['status']): TodoItem => ({ content: `${status} item`, id: status, status })

const subagent = (id: string, status: SubagentProgress['status']): SubagentProgress => ({
  filesRead: [],
  filesWritten: [],
  goal: id,
  id,
  parentId: null,
  startedAt: 0,
  status,
  stream: [],
  taskCount: 1,
  taskIndex: 0,
  updatedAt: 0
})

beforeEach(() => {
  $activeSessionId.set(null)
  $attentionSessionIds.set([])
  $selectedStoredSessionId.set(null)
  $workingSessionIds.set([])
  $todosBySession.set({})
  $subagentsBySession.set({})
  $workstreamMetadata.set({})
})

describe('WORKSTREAM_STATE_META', () => {
  it('covers every Telegram-derived workstream state', () => {
    expect(WORKSTREAM_STATE_META).toEqual({
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
    })

    expect(Object.keys(WORKSTREAM_STATE_META).sort()).toEqual([...WORKSTREAM_STATES].sort())

    for (const state of WORKSTREAM_STATES) {
      expect(WORKSTREAM_STATE_META[state].icon).toBeTruthy()
      expect(WORKSTREAM_STATE_META[state].label).toBeTruthy()
      expect(WORKSTREAM_STATE_META[state].tone).toBeTruthy()
    }
  })
})

describe('deriveWorkstreamActivity', () => {
  it('prioritizes needs-input over running work', () => {
    expect(deriveWorkstreamActivity({ session: session(), isWorking: true, needsInput: true }).state).toBe('warn')
  })

  it('prioritizes failed subagents over active subagents', () => {
    expect(
      deriveWorkstreamActivity({
        session: session(),
        subagents: [{ status: 'running' }, { status: 'failed' }]
      }).state
    ).toBe('blocked')
  })

  it('uses delegate when subagents are still active', () => {
    const activity = deriveWorkstreamActivity({
      session: session(),
      subagents: [{ status: 'queued' }, { status: 'running' }]
    })

    expect(activity.state).toBe('delegate')
    expect(activity.activeSubagentCount).toBe(2)
  })

  it('uses work when todos are active even before a tool event marks the session working', () => {
    const activity = deriveWorkstreamActivity({
      session: session(),
      todos: [{ id: 'a', content: 'Do it', status: 'pending' }]
    })

    expect(activity.state).toBe('work')
    expect(activity.activeTodoCount).toBe(1)
  })

  it('falls back to idle for quiet sessions', () => {
    expect(deriveWorkstreamActivity({ session: session() }).state).toBe('idle')
  })

  it('lets explicit source events surface every state while auto-derivation stays conservative', () => {
    for (const state of WORKSTREAM_STATES) {
      expect(deriveWorkstreamActivity({ explicitState: state, session: session() }).state).toBe(state)
    }
  })

  it('does not count completed or cancelled todos as active sidebar work', () => {
    const activity = deriveWorkstreamActivity({
      session: session(),
      todos: [todo('completed'), todo('cancelled')]
    })

    expect(activity.state).toBe('done')
    expect(activity.activeTodoCount).toBe(0)
    expect(activity.totalTodoCount).toBe(2)
  })

  it('preserves per-session selector identity when other sessions update', () => {
    const activity = $workstreamActivity('s1')
    const first = activity.get()

    $subagentsBySession.set({ s2: [subagent('other', 'running')] })

    expect(activity.get()).toBe(first)

    $todosBySession.set({ s2: [todo('pending')] })

    expect(activity.get()).toBe(first)

    $attentionSessionIds.set(['s2'])

    expect(activity.get()).toBe(first)

    $workingSessionIds.set(['s2'])

    expect(activity.get()).toBe(first)

    $todosBySession.set({ s1: [todo('pending')] })

    expect(activity.get()).not.toBe(first)
    expect(activity.get().activeTodoCount).toBe(1)
  })

  it('resolves live runtime-keyed activity for the selected stored sidebar row', () => {
    $selectedStoredSessionId.set('stored-1')
    $activeSessionId.set('runtime-1')
    $todosBySession.set({ 'runtime-1': [todo('pending')] })
    $subagentsBySession.set({ 'runtime-1': [subagent('agent', 'running')] })

    const activity = $workstreamActivity('stored-1').get()

    expect(activity.sessionId).toBe('stored-1')
    expect(activity.state).toBe('delegate')
    expect(activity.activeTodoCount).toBe(1)
    expect(activity.activeSubagentCount).toBe(1)
  })

  it('lets desktop lifecycle metadata override runtime-derived workstream state', () => {
    $workingSessionIds.set(['stored-1'])
    $workstreamMetadata.set({ 'stored-1': { lifecycle: 'restart_required', updatedAt: 123 } })

    const restartActivity = $workstreamActivity('stored-1').get()

    expect(restartActivity.state).toBe('restart')
    expect(restartActivity.label).toBe('restart required')
    expect(restartActivity.isWorking).toBe(true)

    $workstreamMetadata.set({ 'stored-1': { lifecycle: 'closed', updatedAt: 456 } })

    const closedActivity = $workstreamActivity('stored-1').get()

    expect(closedActivity.state).toBe('close')
    expect(closedActivity.label).toBe('safe to delete')
    expect($workstreamMetadata.get()['stored-1']?.lifecycle).toBe('closed')

    $workstreamMetadata.set({ 'stored-1': { lifecycle: 'safe_delete', updatedAt: 789 } })

    const safeDeleteActivity = $workstreamActivity('stored-1').get()

    expect(safeDeleteActivity.state).toBe('close')
    expect(safeDeleteActivity.label).toBe('safe to delete')
    expect($workstreamMetadata.get()['stored-1']?.lifecycle).toBe('safe_delete')
  })
})
