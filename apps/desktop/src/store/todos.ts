import { atom } from 'nanostores'

import type { TodoItem } from '@/lib/todos'

/**
 * Session-scoped structured task plans emitted by the `todo` tool.
 *
 * Plans are durable operational state, not transient turn decoration. They stay
 * visible until the agent explicitly replaces them or clears the list, and are
 * keyed by runtime session so concurrent chats never overwrite each other.
 */
export const $todosBySession = atom<Record<string, TodoItem[]>>({})

/** Contrib wiring provides the authoritative active-session reconciliation
 * handler. The pane consumes it without importing the controller that mounts
 * the pane, avoiding a circular dependency. */
export type RunBoardRefreshResult = 'applied' | 'superseded' | 'unchanged'

export const $runBoardRefresh = atom<(() => Promise<RunBoardRefreshResult | void>) | null>(null)

export const todoListActive = (todos: readonly TodoItem[]) =>
  todos.some(t => t.status === 'pending' || t.status === 'in_progress')

/** Restore the latest structured plan exactly as recorded. Turn completion,
 * errors, reloads, and compaction must not silently discard work the agent has
 * not explicitly replaced or cleared. */
export function todosForHydration(todos: readonly TodoItem[] | null): TodoItem[] | null {
  return todos ? [...todos] : null
}

export function setSessionTodos(sid: string, todos: TodoItem[]) {
  if (!sid) {
    return
  }

  if (todos.length === 0) {
    clearSessionTodos(sid)

    return
  }

  $todosBySession.set({ ...$todosBySession.get(), [sid]: todos })
}

/** Apply persisted board state only if the session's live plan is still the
 * exact snapshot observed when the request began. A live todo tool update wins
 * over an older response, and a transcript with no todo call cannot erase a
 * renderer-only plan that may not have persisted yet. */
export function applyHydratedSessionTodos(
  sid: string,
  expected: TodoItem[] | undefined,
  hydrated: TodoItem[] | null
): RunBoardRefreshResult {
  if ($todosBySession.get()[sid] !== expected) {
    return 'superseded'
  }

  if (hydrated === null) {
    return 'unchanged'
  }

  setSessionTodos(sid, hydrated)

  return 'applied'
}

/** Re-fetch one session's persisted board without exposing any run/resume API.
 * The request is invalidated if foreground identity changes before it settles. */
export async function refreshSessionTodos(
  sid: string,
  load: () => Promise<TodoItem[] | null>,
  isCurrent: () => boolean
): Promise<RunBoardRefreshResult> {
  const expected = $todosBySession.get()[sid]
  const hydrated = await load()

  if (!isCurrent()) {
    return 'superseded'
  }

  return applyHydratedSessionTodos(sid, expected, hydrated)
}

export function clearSessionTodos(sid: string) {
  const map = $todosBySession.get()

  if (!(sid in map)) {
    return
  }

  const { [sid]: _drop, ...rest } = map
  $todosBySession.set(rest)
}
