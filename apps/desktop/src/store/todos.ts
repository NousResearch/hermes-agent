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
 * - stored-session hydration (desktop-controller) — but only when the list is
 *   still in flight, so reopening an old chat doesn't pin its finished plan
 *   above the composer forever.
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

// Decide which todo list to restore when rehydrating a session from stored
// history. Rehydration runs *after* a turn completes, so an active list (last
// item still pending/in_progress) is stale — the turn ended without a final
// `todo` update — and must NOT be re-pinned (that would undo the turn-end
// clear and, because it's read back from history, resurrect on restart). Only
// a finished list is restored, so its short linger shows the last checkmark.
// Returns null when there's nothing to restore (caller should clear).
export function todosForHydration(todos: readonly TodoItem[] | null): TodoItem[] | null {
  return todos && !todoListActive(todos) ? [...todos] : null
}

// Once a list finishes (every item completed/cancelled), the final state
// lingers just long enough to see the last checkmark land, then the group
// drops out of the stack on its own.
const FINISHED_LINGER_MS = 4_000
const clearTimers = keyedTimeouts()

function acceptRevision(sid: string, revision?: null | number): boolean {
  const revisions = $todoRevisionsBySession.get()
  const current = revisions[sid]

  if (revision == null) {
    return current == null
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
  const visible = running ? todos : todosForHydration(todos)

  if (visible !== null) {
    setSessionTodos(sid, visible, revision)
  } else if (acceptRevision(sid, revision)) {
    dropSessionTodos(sid, false)
  }
}
