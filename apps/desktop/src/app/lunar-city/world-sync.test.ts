import { describe, expect, it, vi } from 'vitest'

import type { WorldCursorState } from '@/store/lunar-city'

import type { WorldEvent } from './world-events'
import { bindWorldSources, reconcileWorldSnapshot, type WorldSourceDoors, worldSourceScopeKey } from './world-sync'

const cursors: WorldCursorState = { bySource: {}, dismissedRecapIds: [], lastOpenedAt: null }

function event(overrides: Partial<WorldEvent> = {}): WorldEvent {
  return {
    actionKinds: ['inspect'],
    facts: {},
    id: 'kanban:41',
    kind: 'task.blocked',
    occurredAt: 41,
    receivedAt: 41,
    scope: 'task',
    severity: 'warning',
    source: 'kanban',
    sourceRef: { board: 'main', taskId: 'task-7' },
    title: 'Blocked',
    transition: true,
    ...overrides
  }
}

describe('world synchronization', () => {
  it('does not replay an event already acknowledged for the same board', () => {
    const previous = { ...cursors, bySource: { 'kanban:main': 'kanban:41' } }
    const result = reconcileWorldSnapshot({ tasks: [] }, [event()], previous, 100)

    expect(result.projection.transitions).toEqual([])
    expect(result.projection.recentEvents).toHaveLength(1)
  })

  it('keeps event cursors isolated by board and source', () => {
    const result = reconcileWorldSnapshot(
      { tasks: [] },
      [event({ sourceRef: { board: 'other', taskId: 'task-7' } })],
      { ...cursors, bySource: { 'kanban:main': 'kanban:41' } },
      100
    )

    expect(worldSourceScopeKey(event({ sourceRef: { board: 'other' } }))).toBe('kanban:other')
    expect(result.projection.transitions).toHaveLength(1)
  })

  it('derives current conditions from the reopened snapshot', () => {
    const result = reconcileWorldSnapshot(
      {
        tasks: [
          { id: 'blocked', status: 'blocked', title: 'Fix auth' },
          { id: 'done', status: 'done', title: 'Ship release' }
        ]
      },
      [],
      cursors,
      100
    )

    expect(result.projection.conditions).toHaveLength(2)
    expect(result.projection.conditions.map(condition => condition.kind)).toEqual(['task.blocked', 'task.completed'])
  })

  it('disposes every registered source when the world closes', () => {
    const closeKanban = vi.fn()
    const closeNotices = vi.fn()

    const doors: WorldSourceDoors = {
      kanban: listener => {
        expect(listener).toBeTypeOf('function')

        return closeKanban
      },
      notices: listener => {
        expect(listener).toBeTypeOf('function')

        return closeNotices
      }
    }

    const publish = vi.fn()
    const dispose = bindWorldSources(doors, { getCursors: () => cursors, publish })

    dispose()

    expect(closeKanban).toHaveBeenCalledOnce()
    expect(closeNotices).toHaveBeenCalledOnce()
    expect(publish).not.toHaveBeenCalled()
  })
})
