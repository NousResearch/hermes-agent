import { atom, computed } from 'nanostores'
import { batch } from 'nanostores'

import { keyedTimeouts } from '@/lib/keyed-timeouts'
import { stableRecord } from '@/lib/stable-array'
import type { TodoItem, TodoSnapshot } from '@/lib/todos'

import { $sessions, lineageAliases } from './session'
import { $sessionStates } from './session-states'

/**
 * Live todo list per runtime session, rendered by the composer status stack
 * (the inline transcript panel is gone). Fed from three places:
 *
 * - live `todo` tool events (use-message-stream) — display lists
 * - authoritative snapshots (`todo.snapshot` RPC, `todo.updated` event,
 *   revision-stamped `tool.complete` payloads) via setSessionTodoSnapshot
 * - stored-session hydration (desktop-controller) — but only when the list is
 *   still in flight, so reopening an old chat doesn't pin its finished plan
 *   above the composer forever.
 */
export const $todosBySession = atom<Record<string, TodoItem[]>>({})

/** Revision + generation authority per session, published in the same batch
 *  as the display list. A session in here has human mutation authority
 *  (Mark done/Reopen controls enabled); a session only in $todosBySession is
 *  display-only (controls disabled, "Syncing task status"). */
export interface TodoMutationAuthority {
  generation: number
  revision: number
}

export const $sessionTodoSnapshots = atom<Record<string, TodoMutationAuthority | null>>({})

/** Mutation authority for a session — null means display-only. */
export const todoSnapshotAuthority = (sid: string): TodoMutationAuthority | null =>
  $sessionTodoSnapshots.get()[sid] ?? null

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
// drops out of the stack on its own. This is the LEGACY display-only path:
// an authoritative snapshot never schedules it — a user-completed plan must
// stay up so the Reopen control doesn't vanish.
const FINISHED_LINGER_MS = 4_000
const clearTimers = keyedTimeouts()

const publishTodos = (sid: string, todos: TodoItem[]) => {
  $todosBySession.set({ ...$todosBySession.get(), [sid]: todos })
}

const publishAuthority = (sid: string, authority: TodoMutationAuthority | null) => {
  $sessionTodoSnapshots.set({ ...$sessionTodoSnapshots.get(), [sid]: authority })
}

export function setSessionTodos(sid: string, todos: TodoItem[]) {
  if (!sid) {
    return
  }

  clearTimers.cancel(sid)

  // Display list and display-only authority move as one pair: a plain tool
  // list carries no revision, so whatever authority this session had is gone
  // (the controls must disable rather than CAS against a revision we can't
  // see). batch() keeps subscribers from observing authority without list.
  batch(() => {
    publishTodos(sid, todos)
    publishAuthority(sid, null)
  })

  if (!todoListActive(todos)) {
    clearTimers.schedule(sid, FINISHED_LINGER_MS, () => clearSessionTodos(sid))
  }
}

/** Adopt an authoritative full snapshot (todo.snapshot RPC result /
 *  todo.update_status response / todo.updated event / revision-stamped
 *  tool.complete payload). Higher generation wins; an equal generation may
 *  replace in place (an exact refetch of the same plan). Never schedules the
 *  finished-linger clear — terminal authoritative lists stay mounted so the
 *  human controls remain. */
export function setSessionTodoSnapshot(snapshot: TodoSnapshot) {
  if (!snapshot.session_id) {
    return
  }

  const sid = snapshot.session_id
  const current = $sessionTodoSnapshots.get()[sid]

  if (current && snapshot.generation < current.generation) {
    return
  }

  clearTimers.cancel(sid)

  batch(() => {
    publishTodos(sid, snapshot.todos)
    publishAuthority(sid, { generation: snapshot.generation, revision: snapshot.revision })
  })
}

export function clearSessionTodos(sid: string) {
  clearTimers.cancel(sid)

  const map = $todosBySession.get()
  const snapshots = $sessionTodoSnapshots.get()

  const nextSnapshots = { ...snapshots }
  delete nextSnapshots[sid]

  batch(() => {
    if (sid in map) {
      const { [sid]: _drop, ...rest } = map
      $todosBySession.set(rest)
    }

    if (sid in snapshots) {
      $sessionTodoSnapshots.set(nextSnapshots)
    }
  })
}

// Drop a still-active todo list (any pending/in_progress item) — used at turn
// end, when an unfinished list means the turn stopped without a final `todo`
// update, so the "Tasks N/M" panel would otherwise stay pinned above the
// composer forever. An AUTHORITATIVE list is exempt: the live snapshot is the
// store's own truth (the heuristic can't know a pending item is real), so
// turn-end cleanup never races it away. A finished list is also left alone —
// its short linger shows the last checkmark landing, and an authoritative one
// stays up (the user's Mark done decision is the source of truth; clearing it
// would yank the Reopen control).
export function clearActiveSessionTodos(sid: string) {
  const todos = $todosBySession.get()[sid]

  if (!todos || !todoListActive(todos)) {
    return
  }

  if (todoSnapshotAuthority(sid) !== null) {
    return
  }

  clearSessionTodos(sid)
}