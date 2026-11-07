/**
 * Board deltas. The board is one query of the whole thing (thousands of cards), and nearly every
 * poll and events frame changed a handful of them, so the server answers a request that carries the
 * held board's cursor (`since`) and etag with what moved, or with a few bytes saying nothing did.
 * This module is the client half of that contract: the request plan, and the merge.
 *
 * Contract, as answered by `GET /board` on a backend that supports it:
 *   no `since`  → a full snapshot, `{columns, tenants, assignees, latest_event_id, now, etag}`
 *   `since`+`etag` → `{delta: true, unchanged: true, ...}` or
 *                    `{delta: true, cards, column_order, removed, tenants, assignees, ...}`
 * A backend that does not know `since`/`etag` ignores them and answers a normal snapshot with no
 * `delta` key and no `etag`; the client never asks for a delta without an etag in hand, and treats
 * any response without `delta: true` as a full replace, which is what keeps an older backend working.
 *
 * Pure functions only: no React, no query cache, no clock. `api.ts` does the IO.
 */

import type { KanbanBoard, KanbanBoardDelta, KanbanTask } from './types'

/** The held board is re-fetched in full this often, whatever the deltas say. A delta stream can miss
 *  a change only through a server bug; this bounds how long a mistake can survive on screen. */
export const FULL_RESYNC_MS = 15 * 60_000

/** The cursor a request may carry, or null for "ask for a full snapshot". */
export interface DeltaCursor {
  etag: string
  since: number
}

function isCard(value: unknown): value is KanbanTask {
  return typeof value === 'object' && value !== null && typeof (value as KanbanTask).id === 'string'
}

/** Does this response describe a delta (rather than a full snapshot)? */
export function isBoardDelta(value: unknown): value is KanbanBoardDelta {
  return typeof value === 'object' && value !== null && (value as { delta?: unknown }).delta === true
}

/**
 * The `since`/`etag` pair to send, or null to ask for a full snapshot. A held board is only a base
 * for a delta when it came from a full fetch (it has an etag and the time of that fetch), still has
 * a cursor, and that full fetch is younger than FULL_RESYNC_MS. A clock that moved backwards is
 * treated the same as age: ask again in full rather than trust the span.
 */
export function deltaCursor(held: KanbanBoard | undefined, now: number): DeltaCursor | null {
  if (!held) {
    return null
  }

  const { etag, fullFetchedAt, latest_event_id: since } = held

  if (typeof etag !== 'string' || etag === '' || typeof since !== 'number') {
    return null
  }

  if (typeof fullFetchedAt !== 'number') {
    return null
  }

  const age = now - fullFetchedAt

  if (!Number.isFinite(age) || age < 0 || age > FULL_RESYNC_MS) {
    return null
  }

  return { etag, since }
}

/** A full snapshot, stamped for the render: every card carries the payload clock (see arcState) and
 *  the board records when the full fetch happened, which is what `deltaCursor` ages. */
export function stampSnapshot(board: KanbanBoard, now: number): KanbanBoard {
  return {
    ...board,
    columns: (board.columns ?? []).map(column => ({
      ...column,
      tasks: (column.tasks ?? []).map(task => ({ ...task, heartbeat_now: board.now }))
    })),
    fullFetchedAt: now
  }
}

/** id → card, so the rebuild is one lookup per row instead of a scan per column. */
function indexCards(board: KanbanBoard): Map<string, KanbanTask> {
  const index = new Map<string, KanbanTask>()

  for (const column of board.columns ?? []) {
    for (const task of column.tasks ?? []) {
      index.set(task.id, task)
    }
  }

  return index
}

/**
 * Merge a delta into the held board. Returns the new board, or null when the client is out of sync
 * (the delta names an id it does not hold) and the caller must refetch a full snapshot at once.
 *
 * The rule, exactly as the server half states it: start from the held cards, delete `removed`, set
 * every `cards` entry by id, then rebuild every column from `column_order`. Cards the delta did not
 * send keep their previous object identity, so a memoized card does not repaint; a column whose ids
 * all resolve to the same cards in the same order keeps its column object too.
 *
 * Not refreshed: the cards' server `age` block. The desktop never reads it (it computes elapsed time
 * from created_at / started_at on its own clock), so re-aging every untouched card would cost a new
 * object per card for a field nothing paints.
 */
export function mergeBoardDelta(held: KanbanBoard, delta: KanbanBoardDelta): KanbanBoard | null {
  if (delta.unchanged === true) {
    return adoptClock(held, delta)
  }

  const cards = Array.isArray(delta.cards) ? delta.cards.filter(isCard) : []

  if (!Array.isArray(delta.column_order)) {
    // A delta in motion always carries the full column order, even with no changed cards (a removal
    // or a reorder sends no `cards`). Without it there is nothing to rebuild from.
    return null
  }

  const nextCards = indexCards(held)

  for (const id of delta.removed ?? []) {
    nextCards.delete(id)
  }

  for (const task of cards) {
    nextCards.set(task.id, { ...task, heartbeat_now: delta.now })
  }

  const columns: KanbanBoard['columns'] = []

  for (const { ids, name } of delta.column_order) {
    const tasks: KanbanTask[] = []

    for (const id of ids ?? []) {
      const task = nextCards.get(id)

      if (!task) {
        return null
      }

      tasks.push(task)
    }

    const previous = held.columns.find(column => column.name === name)

    columns.push(previous && sameTasks(previous.tasks, tasks) ? previous : { name, tasks })
  }

  return {
    assignees: Array.isArray(delta.assignees) ? delta.assignees : held.assignees,
    columns,
    etag: typeof delta.etag === 'string' ? delta.etag : held.etag,
    fullFetchedAt: held.fullFetchedAt,
    latest_event_id: delta.latest_event_id,
    now: delta.now,
    tenants: Array.isArray(delta.tenants) ? delta.tenants : held.tenants
  }
}

function sameTasks(previous: KanbanTask[], next: KanbanTask[]): boolean {
  if (previous.length !== next.length) {
    return false
  }

  for (let i = 0; i < next.length; i += 1) {
    if (previous[i] !== next[i]) {
      return false
    }
  }

  return true
}

/**
 * "Nothing changed": keep the data the client holds and adopt only the new etag, cursor and clock.
 * One exception, and it is required for a merged board to read like a fresh one: a RUNNING card ages
 * its heartbeat against the payload clock (arcState), and a quiet worker sends no events, so a
 * running card's stamp has to move with `now` or it would read stale on a fresh snapshot and live on
 * a delta. Only running cards are restamped; every column and card object otherwise keeps identity.
 */
function adoptClock(held: KanbanBoard, delta: KanbanBoardDelta): KanbanBoard {
  let touched = false

  const columns = held.columns.map(column => {
    if (!column.tasks.some(task => task.status === 'running' && task.heartbeat_now !== delta.now)) {
      return column
    }

    touched = true

    return {
      ...column,
      tasks: column.tasks.map(task =>
        task.status === 'running' && task.heartbeat_now !== delta.now ? { ...task, heartbeat_now: delta.now } : task
      )
    }
  })

  return {
    ...held,
    columns: touched ? columns : held.columns,
    etag: typeof delta.etag === 'string' ? delta.etag : held.etag,
    latest_event_id: delta.latest_event_id,
    now: delta.now
  }
}
