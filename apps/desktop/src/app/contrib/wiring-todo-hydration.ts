import { latestSessionTodos } from '@/lib/todos'
import { $todoContinuationsBySession, clearSessionTodos, setSessionTodos, todosForHydration } from '@/store/todos'

/** Apply stored transcript todo state for the exact runtime being hydrated. */
export function hydrateSessionTodos(runtimeSessionId: string, messages: readonly { parts?: unknown }[]): void {
  const restored = todosForHydration(latestSessionTodos(messages), $todoContinuationsBySession.get()[runtimeSessionId])

  if (restored) {
    setSessionTodos(runtimeSessionId, restored)
  } else {
    clearSessionTodos(runtimeSessionId)
  }
}
