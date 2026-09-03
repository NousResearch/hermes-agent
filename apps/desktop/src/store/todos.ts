import { atom, computed } from 'nanostores'

import { keyedTimeouts } from '@/lib/keyed-timeouts'
import { stableRecord } from '@/lib/stable-array'
import { parseTodoRevision, parseTodos, type TodoItem } from '@/lib/todos'

import { $sessions, lineageAliases } from './session'
import { $sessionStates } from './session-states'

/**
 * Live todo list per runtime session, rendered by the composer status stack
 * (the inline transcript panel is gone). Fed from two places:
 *
 * - live `todo` tool events (use-message-stream)
 * - stored-session hydration (desktop-controller) — only while a session is
 *   still running, so reopening an old chat never re-pins its historic plan.
 */
export const $todosBySession = atom<Record<string, TodoItem[]>>({})
export const $todoRevisionsBySession = atom<Record<string, number>>({})

export const todoListActive = (todos: readonly TodoItem[]) =>
  todos.some(t => t.status === 'pending' || t.status === 'in_progress')

let todoProgress: Readonly<Record<string, string>> = {}

/** Live "X/Y" per STORED session id, for the sidebar's inbox cards. The live
 *  map keys on runtime ids; this projects through the same storedSessionId +
 *  lineage-alias fallback as the working/attention projections, so the card
 *  finds its count under the id the sidebar knows. Cancelled items don't
 *  count toward either side of the fraction. Values are the rendered "X/Y"
 *  string — primitives, so stableRecord can suppress no-op emits. */
export const $todoProgressBySession = computed(
  [$todosBySession, $sessionStates, $sessions],
  (todosMap, states, sessions) => {
    const next: Record<string, string> = {}

    for (const [runtimeId, todos] of Object.entries(todosMap)) {
      const counted = todos.filter(t => t.status !== 'cancelled')

      if (counted.length === 0) {
        continue
      }

      const progress = `${counted.filter(t => t.status === 'completed').length}/${counted.length}`

      for (const alias of lineageAliases(states[runtimeId]?.storedSessionId ?? runtimeId, sessions)) {
        next[alias] = progress
      }
    }

    return (todoProgress = stableRecord(todoProgress, next))
  }
)

// Stored todo snapshots are transcript history, not a new progress event. The
// live event already shows the final checkmark, so restoring either an active
// or completed list after the turn ends would make the composer pop back open.
// Returns null so the caller clears the stale presentation state.
export function todosForHydration(_todos: readonly TodoItem[] | null): TodoItem[] | null {
  return null
}

// Once a list finishes (every item completed/cancelled), the final state
// lingers just long enough to see the last checkmark land, then the group
// drops out of the stack on its own.
const FINISHED_LINGER_MS = 4_000
const clearTimers = keyedTimeouts()

function acceptRevision(sid: string, revision?: null | number): boolean {
  const revisions = $todoRevisionsBySession.get()
  const current = revisions[sid]

  // tool.start has no revision. Apply the merge locally and leave the
  // watermark alone so a later todo.updated / tool.complete can still win.
  if (revision == null) {
    return true
  }

  if (current != null && revision < current) {
    return false
  }

  if (current !== revision) {
    $todoRevisionsBySession.set({ ...revisions, [sid]: revision })
  }

  return true
}

export function setSessionTodos(sid: string, todos: TodoItem[], revision?: null | number) {
  if (!sid) {
    return
  }

  const currentRevision = $todoRevisionsBySession.get()[sid]
  const previous = $todosBySession.get()[sid]

  // A completed list remains in the backend snapshot after its short UI linger.
  // Reading it again must not reopen the composer status group or reset its timer.
  // Keep a same-revision completion only when it replaces a still-visible active
  // list, which is the final checkmark transition for that live plan.
  if (
    !todoListActive(todos) &&
    revision != null &&
    currentRevision === revision &&
    !todoListActive(previous ?? [])
  ) {
    return
  }

  if (!acceptRevision(sid, revision)) {
    return
  }

  clearTimers.cancel(sid)
  $todosBySession.set({ ...$todosBySession.get(), [sid]: todos })

  if (!todoListActive(todos)) {
    clearTimers.schedule(sid, FINISHED_LINGER_MS, () => dropSessionTodos(sid, false))
  }
}

function dropSessionTodos(sid: string, forgetRevision: boolean) {
  clearTimers.cancel(sid)

  const map = $todosBySession.get()

  if (sid in map) {
    const { [sid]: _drop, ...rest } = map
    $todosBySession.set(rest)
  }

  if (forgetRevision) {
    const revisions = $todoRevisionsBySession.get()

    if (sid in revisions) {
      const { [sid]: _drop, ...rest } = revisions
      $todoRevisionsBySession.set(rest)
    }
  }
}

export function clearSessionTodos(sid: string) {
  dropSessionTodos(sid, true)
}

// Drop a still-active todo list (any pending/in_progress item) — used at turn
// end, when an unfinished list means the turn stopped without a final `todo`
// update, so the "Tasks N/M" panel would otherwise stay pinned above the
// composer forever. A finished list is left untouched so its short linger
// still shows the last checkmark landing.
export function clearActiveSessionTodos(sid: string) {
  const todos = $todosBySession.get()[sid]

  if (!todos || !todoListActive(todos)) {
    return
  }

  dropSessionTodos(sid, false)
}

/** Apply a session.resume/activate or todo.updated full snapshot. Idle
 * sessions keep the existing stale-active guard; running sessions restore the
 * active plan because the backend has proved that turn is still live. */
export function restoreSessionTodosFromSnapshot(sid: string, snapshot: unknown, running: boolean) {
  const todos = parseTodos(snapshot)

  if (!sid || todos === null) {
    return
  }

  const revision = parseTodoRevision(snapshot)

  // An unused store serializes as {todos: [], revision: 0}. That is not a
  // real snapshot. Applying it would stamp watermark 0 and leave an empty
  // list in the map.
  if (todos.length === 0 && (revision == null || revision === 0)) {
    return
  }

  const visible = running ? todos : todosForHydration(todos)

  if (visible !== null) {
    setSessionTodos(sid, visible, revision)
  } else if (acceptRevision(sid, revision)) {
    dropSessionTodos(sid, false)
  }
}
