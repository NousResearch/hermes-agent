import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $tasksBySession,
  activeAgentTaskCount,
  handleAgentTaskEvent,
  pruneExpiredAgentTasks,
  seedAgentTasks,
  TASK_LINGER_MS,
  upsertAgentTask,
  visibleAgentTasks
} from './tasks'

const listFor = (sid: string) => $tasksBySession.get()[sid] ?? []

// Registry timestamps are epoch seconds; tests mock Date.now() at 2_000_000ms
// so a started_at of 1_990 reads as "10s ago".
const startedSnapshot = (overrides: Record<string, unknown> = {}) => ({
  task_id: 'sa-0-abcd1234',
  intent: 'web_research',
  goal: 'Find the release notes',
  status: 'running',
  started_at: 1_990,
  tool_count: 0,
  session_id: 's1',
  ...overrides
})

describe('agent task store', () => {
  beforeEach(() => {
    $tasksBySession.set({})
    vi.spyOn(Date, 'now').mockReturnValue(2_000_000)
  })

  afterEach(() => vi.restoreAllMocks())

  it('adds a task on task.started and tracks progress fields', () => {
    handleAgentTaskEvent({ payload: startedSnapshot(), type: 'task.started' })

    expect(listFor('s1')).toHaveLength(1)
    expect(listFor('s1')[0]).toMatchObject({
      goal: 'Find the release notes',
      id: 'sa-0-abcd1234',
      intent: 'web_research',
      startedAt: 1_990_000,
      status: 'running'
    })
    expect(activeAgentTaskCount(listFor('s1'))).toBe(1)

    upsertAgentTask('s1', startedSnapshot({ last_tool: 'fetch_url', tool_count: 3 }))

    expect(listFor('s1')[0]).toMatchObject({ lastTool: 'fetch_url', toolCount: 3 })
  })

  it('terminalizes on task.completed and ignores late running updates', () => {
    handleAgentTaskEvent({ payload: startedSnapshot(), type: 'task.started' })
    handleAgentTaskEvent({
      payload: startedSnapshot({ error: 'boom', finished_at: 1_999, status: 'failed' }),
      type: 'task.completed'
    })
    handleAgentTaskEvent({ payload: startedSnapshot({ tool_count: 99 }), type: 'task.started' })

    const task = listFor('s1')[0]
    expect(task).toMatchObject({ error: 'boom', finishedAt: 1_999_000, status: 'failed' })
    expect(task?.toolCount).not.toBe(99)
    expect(activeAgentTaskCount(listFor('s1'))).toBe(0)
  })

  it('routes by the snapshot session_id and ignores events without one', () => {
    const consumed = handleAgentTaskEvent({ payload: startedSnapshot({ session_id: undefined }), type: 'task.started' })

    expect(consumed).toBe(true)
    expect($tasksBySession.get()).toEqual({})

    handleAgentTaskEvent({ payload: startedSnapshot({ session_id: undefined }), session_id: 's2', type: 'task.started' })

    expect(listFor('s2')).toHaveLength(1)
    expect(handleAgentTaskEvent({ payload: startedSnapshot(), type: 'message.delta' })).toBe(false)
  })

  it('seeds from task.list, dropping stale running rows and unattributed snapshots', () => {
    upsertAgentTask('s1', startedSnapshot({ task_id: 'stale-running' }))
    upsertAgentTask('s1', startedSnapshot({ finished_at: 1_999, status: 'succeeded', task_id: 'fresh-done' }))

    seedAgentTasks([
      startedSnapshot({ task_id: 'listed-running' }),
      startedSnapshot({ session_id: '', task_id: 'no-session' })
    ])

    expect(listFor('s1').map(task => task.id)).toEqual(['fresh-done', 'listed-running'])
  })

  it('drops terminal rows after the linger window', () => {
    handleAgentTaskEvent({ payload: startedSnapshot(), type: 'task.started' })
    handleAgentTaskEvent({
      payload: startedSnapshot({ finished_at: 1_999, status: 'succeeded' }),
      type: 'task.completed'
    })

    const finishedAtMs = 1_999_000
    expect(visibleAgentTasks($tasksBySession.get(), finishedAtMs + TASK_LINGER_MS - 1)).toHaveLength(1)
    expect(visibleAgentTasks($tasksBySession.get(), finishedAtMs + TASK_LINGER_MS + 1)).toHaveLength(0)

    // The next write through the store also prunes the expired row for good.
    vi.spyOn(Date, 'now').mockReturnValue(finishedAtMs + TASK_LINGER_MS + 1)
    upsertAgentTask('s1', startedSnapshot({ task_id: 'sa-1-ffff0000' }))

    expect(listFor('s1').map(task => task.id)).toEqual(['sa-1-ffff0000'])
  })

  it('orders visible tasks across sessions by start time and can scope to one session', () => {
    upsertAgentTask('s2', startedSnapshot({ session_id: 's2', started_at: 5, task_id: 'later' }))
    upsertAgentTask('s1', startedSnapshot({ started_at: 1, task_id: 'earlier' }))

    expect(visibleAgentTasks($tasksBySession.get(), 2_000_000).map(task => task.id)).toEqual(['earlier', 'later'])
    expect(visibleAgentTasks($tasksBySession.get(), 2_000_000, 's1').map(task => task.id)).toEqual(['earlier'])
  })

  it('prunes expired terminal rows without waiting for another task write', () => {
    upsertAgentTask('s1', startedSnapshot({ finished_at: 1_990, status: 'succeeded', task_id: 'old-done' }))
    upsertAgentTask('s1', startedSnapshot({ finished_at: 1_999, status: 'failed', task_id: 'fresh-done' }))

    expect(pruneExpiredAgentTasks(1_990_000 + TASK_LINGER_MS + 1)).toBe(true)
    expect(listFor('s1').map(task => task.id)).toEqual(['fresh-done'])
    expect(pruneExpiredAgentTasks(1_990_000 + TASK_LINGER_MS + 1)).toBe(false)
  })
})
