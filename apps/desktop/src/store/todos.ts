import { atom } from 'nanostores'

import {
  todoHistoryFromTranscript,
  type TodoHistoryMessage,
  type TodoHistorySnapshot,
  type TodoItem,
  todoPlanSignature
} from '@/lib/todos'

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

/** Transcript-derived task snapshots keyed by runtime session. This atom is
 * updated only at known mutation boundaries, never while plain text streams. */
export const $todoHistoryBySession = atom<Record<string, TodoHistorySnapshot[]>>({})

function sameTodoHistory(a: readonly TodoHistorySnapshot[], b: readonly TodoHistorySnapshot[]): boolean {
  return (
    a.length === b.length &&
    a.every((snapshot, index) => {
      const other = b[index]

      return (
        other !== undefined &&
        snapshot.id === other.id &&
        snapshot.state === other.state &&
        snapshot.timestamp === other.timestamp &&
        snapshot.todos.length === other.todos.length &&
        snapshot.todos.every(
          (todo, todoIndex) =>
            todo.id === other.todos[todoIndex]?.id &&
            todo.content === other.todos[todoIndex]?.content &&
            todo.status === other.todos[todoIndex]?.status
        )
      )
    })
  )
}

export function rebuildSessionTodoHistory(sid: string, messages: readonly TodoHistoryMessage[]) {
  if (!sid) {
    return
  }

  const history = $todoHistoryBySession.get()
  const next = todoHistoryFromTranscript(messages)

  if (history[sid] && sameTodoHistory(history[sid], next)) {
    return
  }

  $todoHistoryBySession.set({ ...history, [sid]: next })
}

export function rebuildResumedSessionTodoHistory(
  runtimeId: string,
  storedSessionId: string,
  messages: readonly TodoHistoryMessage[]
) {
  if (runtimeId !== storedSessionId) {
    clearSessionTodoHistory(storedSessionId)
  }

  rebuildSessionTodoHistory(runtimeId, messages)
}

export function clearSessionTodoHistory(sid: string) {
  const history = $todoHistoryBySession.get()

  if (!(sid in history)) {
    return
  }

  const { [sid]: _drop, ...rest } = history
  $todoHistoryBySession.set(rest)
}

/** Finalize directly from the session's authoritative live todo turn. The
 * separately rendered list may remain during its 4s linger, but only the turn
 * that produced it may commit a snapshot. */
interface LiveTodoTurn {
  ownerId: string
  todos: TodoItem[]
}

const liveTodoTurns = new Map<string, LiveTodoTurn>()

export function finalizeSessionTodoSnapshot(
  sid: string,
  id: string | null | undefined,
  timestamp = Math.floor(Date.now() / 1_000)
) {
  // No turn owner to match (e.g. the session never streamed a `todo` this turn):
  // nothing authoritative to commit. Early-return so callers don't need a
  // synthetic fallback id that could never match a real owner anyway.
  if (!id) {
    return
  }

  const live = liveTodoTurns.get(sid)

  if (!live || live.ownerId !== id) {
    return
  }

  liveTodoTurns.delete(sid)
  const todos = live.todos

  const signature = todoPlanSignature(todos)
  const previous = $todoHistoryBySession.get()[sid] ?? []

  const snapshot: TodoHistorySnapshot = {
    id,
    state: todoListActive(todos) ? 'unfinished' : 'completed',
    timestamp,
    todos: [...todos]
  }

  $todoHistoryBySession.set({
    ...$todoHistoryBySession.get(),
    [sid]: [snapshot, ...previous.filter(item => todoPlanSignature(item.todos) !== signature)]
  })
}

export const todoListActive = (todos: readonly TodoItem[]) =>
  todos.some(t => t.status === 'pending' || t.status === 'in_progress')

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
const clearTimers = new Map<string, ReturnType<typeof setTimeout>>()

function cancelScheduledClear(sid: string) {
  const timer = clearTimers.get(sid)

  if (timer !== undefined) {
    clearTimeout(timer)
    clearTimers.delete(sid)
  }
}

function dismissSessionTodos(sid: string) {
  const map = $todosBySession.get()

  if (!(sid in map)) {
    return
  }

  const { [sid]: _drop, ...rest } = map
  $todosBySession.set(rest)
}

export function clearAllSessionTodoState() {
  for (const timer of clearTimers.values()) {
    clearTimeout(timer)
  }

  clearTimers.clear()
  liveTodoTurns.clear()
  $todosBySession.set({})
  $todoHistoryBySession.set({})
}

export function setSessionTodos(sid: string, todos: TodoItem[], ownerId?: string | null) {
  if (!sid) {
    return
  }

  cancelScheduledClear(sid)
  $todosBySession.set({ ...$todosBySession.get(), [sid]: todos })

  if (ownerId) {
    liveTodoTurns.set(sid, { ownerId, todos: [...todos] })
  }

  if (!todoListActive(todos)) {
    clearTimers.set(
      sid,
      setTimeout(() => {
        clearTimers.delete(sid)
        dismissSessionTodos(sid)
      }, FINISHED_LINGER_MS)
    )
  }
}

export function clearSessionTodos(sid: string) {
  cancelScheduledClear(sid)
  liveTodoTurns.delete(sid)
  dismissSessionTodos(sid)
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

  clearSessionTodos(sid)
}
