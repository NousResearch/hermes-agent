import { describe, expect, it } from 'vitest'

import { deltaCursor, FULL_RESYNC_MS, isBoardDelta, mergeBoardDelta, stampSnapshot } from './board-delta'
import type { KanbanBoard, KanbanBoardDelta, KanbanTask } from './types'

// A delta is only worth sending if applying it leaves the same board a full read would have. These
// tests state that relationship, and the cases where the client must refuse a delta and ask again.

const task = (id: string, status: string, over: Partial<KanbanTask> = {}): KanbanTask => ({
  id,
  status,
  title: `card ${id}`,
  ...over
})

const board = (columns: Record<string, KanbanTask[]>, over: Partial<KanbanBoard> = {}): KanbanBoard => ({
  assignees: [],
  columns: Object.entries(columns).map(([name, tasks]) => ({ name, tasks })),
  etag: 'e1',
  latest_event_id: 10,
  now: 1_000,
  tenants: [],
  ...over
})

const ids = (b: KanbanBoard) => b.columns.map(column => [column.name, column.tasks.map(t => t.id)])

describe('mergeBoardDelta', () => {
  it('leaves the board a full snapshot of the new state would have', () => {
    const before = stampSnapshot(
      board({
        todo: [task('a', 'todo'), task('b', 'todo')],
        running: [task('c', 'running', { last_heartbeat_at: 990 })],
        done: [task('d', 'done'), task('e', 'done')]
      }),
      0
    )

    // b starts running, c is archived away, f arrives, e is untouched.
    const after = board(
      {
        todo: [task('a', 'todo'), task('f', 'todo')],
        running: [task('b', 'running')],
        done: [task('d', 'done'), task('e', 'done')]
      },
      { etag: 'e2', latest_event_id: 14, now: 1_060 }
    )

    const delta: KanbanBoardDelta = {
      cards: [task('b', 'running'), task('f', 'todo')],
      column_order: after.columns.map(column => ({ ids: column.tasks.map(t => t.id), name: column.name })),
      delta: true,
      etag: 'e2',
      latest_event_id: 14,
      now: 1_060,
      removed: ['c']
    }

    const merged = mergeBoardDelta(before, delta)

    expect(merged).not.toBeNull()
    expect(ids(merged!)).toEqual(ids(after))
    expect(merged).toMatchObject({ etag: 'e2', latest_event_id: 14, now: 1_060 })
    // Cards the delta did not send are the same objects, so a memoized card does not repaint.
    expect(merged!.columns[2]).toBe(before.columns[2])
    expect(merged!.columns[0].tasks[0]).toBe(before.columns[0].tasks[0])
  })

  it('refuses a delta whose column order names a card the client does not hold', () => {
    const held = stampSnapshot(board({ todo: [task('a', 'todo')] }), 0)

    const delta: KanbanBoardDelta = {
      column_order: [{ ids: ['a', 'ghost'], name: 'todo' }],
      delta: true,
      latest_event_id: 11,
      now: 1_001
    }

    expect(mergeBoardDelta(held, delta)).toBeNull()
  })

  it('treats "nothing changed" as the held data with a fresh cursor, and keeps a quiet worker current', () => {
    const held = stampSnapshot(
      board({ running: [task('c', 'running', { last_heartbeat_at: 990 })], todo: [task('a', 'todo')] }),
      0
    )

    const merged = mergeBoardDelta(held, { delta: true, etag: 'e1', latest_event_id: 10, now: 1_030, unchanged: true })

    expect(merged).toMatchObject({ etag: 'e1', now: 1_030 })
    // The running card ages its heartbeat against the payload clock, so it moves with `now`...
    expect(merged!.columns[0].tasks[0].heartbeat_now).toBe(1_030)
    // ...and nothing else is rebuilt.
    expect(merged!.columns[1]).toBe(held.columns[1])
  })
})

describe('deltaCursor', () => {
  const held = stampSnapshot(board({ todo: [task('a', 'todo')] }), 5_000)

  it('asks for a delta only against a board that came from a recent full read with an etag', () => {
    expect(deltaCursor(held, 5_000 + 1_000)).toEqual({ etag: 'e1', since: 10 })
    expect(deltaCursor(undefined, 5_000)).toBeNull()
    expect(deltaCursor({ ...held, etag: undefined }, 5_000)).toBeNull()
    expect(deltaCursor({ ...held, fullFetchedAt: undefined }, 5_000)).toBeNull()
  })

  it('goes back to a full read once the last one is old, or the clock ran backwards', () => {
    expect(deltaCursor(held, 5_000 + FULL_RESYNC_MS + 1)).toBeNull()
    expect(deltaCursor(held, 4_000)).toBeNull()
  })
})

describe('isBoardDelta', () => {
  it('is true only for a response that says it is a delta', () => {
    expect(isBoardDelta({ delta: true })).toBe(true)
    expect(isBoardDelta(board({}))).toBe(false)
    expect(isBoardDelta(null)).toBe(false)
  })
})
