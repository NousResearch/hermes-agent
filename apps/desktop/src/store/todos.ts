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

export function clearSessionTodos(sid: string) {
  const map = $todosBySession.get()

  if (!(sid in map)) {
    return
  }

  const { [sid]: _drop, ...rest } = map
  $todosBySession.set(rest)
}
