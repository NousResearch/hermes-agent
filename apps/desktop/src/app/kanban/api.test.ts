import { describe, expect, it } from 'vitest'

import { columnMap, KANBAN_COLUMNS, statusLabel, type KanbanBoard } from './api'

describe('kanban columnMap', () => {
  it('fills every known column even when board is empty', () => {
    const map = columnMap(null)
    for (const name of KANBAN_COLUMNS) {
      expect(map[name]).toEqual([])
    }
  })

  it('places tasks under their column names', () => {
    const board: KanbanBoard = {
      columns: [
        { name: 'todo', tasks: [{ id: 'a', title: 'A', status: 'todo' }] },
        { name: 'ready', tasks: [{ id: 'b', title: 'B', status: 'ready' }] }
      ]
    }
    const map = columnMap(board)
    expect(map.todo).toHaveLength(1)
    expect(map.ready[0]?.id).toBe('b')
    expect(map.done).toEqual([])
  })
})

describe('statusLabel', () => {
  it('capitalizes status keys', () => {
    expect(statusLabel('ready')).toBe('Ready')
    expect(statusLabel('')).toBe('')
  })
})
