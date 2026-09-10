import { describe, expect, it } from 'vitest'

import { applyHeartbeatEvents, eventsNeedBoardRefresh } from './api'
import type { KanbanBoard } from './types'

const board = (): KanbanBoard => ({
  columns: [
    {
      name: 'running',
      tasks: [
        { id: 't_running', status: 'running', title: 'Working', last_heartbeat_at: 10 },
        { id: 't_other', status: 'running', title: 'Other', last_heartbeat_at: 11 }
      ]
    }
  ],
  tenants: [],
  assignees: [],
  latest_event_id: 20,
  now: 20
})

describe('Kanban event-driven cache updates', () => {
  it('does not schedule a whole-board refresh for heartbeat-only frames', () => {
    expect(
      eventsNeedBoardRefresh([
        { id: 21, task_id: 't_running', kind: 'heartbeat', created_at: 30, payload: null },
        { id: 22, task_id: 't_other', kind: 'heartbeat', created_at: 31, payload: null }
      ])
    ).toBe(false)
  })

  it('does not reload the board for a repeated respawn guard observation', () => {
    expect(
      eventsNeedBoardRefresh([
        {
          id: 23,
          task_id: 't_running',
          kind: 'respawn_guarded',
          created_at: 32,
          payload: { reason: 'active_pr' }
        }
      ])
    ).toBe(false)
  })

  it('patches heartbeat liveness and cursor into the cached board', () => {
    const updated = applyHeartbeatEvents(board(), [
      { id: 22, task_id: 't_running', kind: 'heartbeat', created_at: 31, payload: null }
    ])

    expect(updated.columns[0].tasks[0].last_heartbeat_at).toBe(31)
    expect(updated.columns[0].tasks[1].last_heartbeat_at).toBe(11)
    expect(updated.latest_event_id).toBe(22)
    expect(updated.now).toBe(31)
  })

  it('requests a refresh for any event that can change board state', () => {
    expect(
      eventsNeedBoardRefresh([
        { id: 21, task_id: 't_running', kind: 'heartbeat', created_at: 30, payload: null },
        { id: 22, task_id: 't_running', kind: 'status', created_at: 31, payload: { status: 'done' } }
      ])
    ).toBe(true)
  })
})
