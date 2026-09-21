import { parseTodoSnapshot, type TodoSnapshot, type TodoStatus } from '@/lib/todos'

export type TodoHumanStatus = 'completed' | 'pending'
export type TodoMutationFailureKind = 'missing' | 'session' | 'stale' | 'unavailable'

export interface TodoGatewayRequest {
  <T>(method: string, params?: Record<string, unknown>): Promise<T>
}

export interface TodoMutationDeps {
  applySnapshot: (snapshot: TodoSnapshot) => void
  getSnapshot: (sessionId: string) => null | TodoSnapshot
  optimisticStatus: (sessionId: string, itemId: string, status: TodoHumanStatus) => boolean
  request: TodoGatewayRequest
}

export interface TodoMutationInput {
  content: string
  expectedGeneration: number
  expectedRevision: number
  itemId: string
  sessionId: string
  status: TodoHumanStatus
}

export class TodoMutationFailure extends Error {
  constructor(
    readonly kind: TodoMutationFailureKind,
    message: string,
    options?: ErrorOptions
  ) {
    super(message, options)
    this.name = 'TodoMutationFailure'
  }
}

export const humanTodoTarget = (status: TodoStatus): TodoHumanStatus =>
  status === 'completed' || status === 'cancelled' ? 'pending' : 'completed'

export const todoGatewayErrorCode = (error: unknown): number | null => {
  if (!error || typeof error !== 'object') {
    return null
  }

  const code = (error as { code?: unknown }).code

  return typeof code === 'number' && Number.isInteger(code) ? code : null
}

const errorSnapshot = (error: unknown, sessionId: string): TodoSnapshot | null => {
  if (!error || typeof error !== 'object') {
    return null
  }

  const snapshot = parseTodoSnapshot((error as { data?: unknown }).data)

  return snapshot?.session_id === sessionId ? snapshot : null
}

const failureKind = (error: unknown): TodoMutationFailureKind => {
  const code = todoGatewayErrorCode(error)

  if (code === 4096) {
    return 'stale'
  }

  if (code === 4044) {
    return 'missing'
  }

  if (code === 4001) {
    return 'session'
  }

  return 'unavailable'
}

const errorMessage = (error: unknown): string =>
  error instanceof Error && error.message.trim() ? error.message : 'Task update failed'

export function createTodoMutationController(deps: TodoMutationDeps) {
  const pending = new Map<string, Promise<TodoSnapshot>>()

  const run = (input: TodoMutationInput): Promise<TodoSnapshot> => {
    const key = `${input.sessionId}\u0000${input.itemId}`
    const existing = pending.get(key)

    if (existing) {
      return existing
    }

    const before = deps.getSnapshot(input.sessionId)

    if (!before) {
      return Promise.reject(new TodoMutationFailure('unavailable', 'Task status is still syncing'))
    }

    if (before.generation !== input.expectedGeneration || before.revision !== input.expectedRevision) {
      return Promise.reject(new TodoMutationFailure('stale', 'Tasks changed while confirmation was open'))
    }

    const selected = before.todos.find(todo => todo.id === input.itemId)

    if (!selected || selected.content !== input.content) {
      return Promise.reject(new TodoMutationFailure('missing', 'That task is no longer in the current plan'))
    }

    if (!deps.optimisticStatus(input.sessionId, input.itemId, input.status)) {
      return Promise.reject(new TodoMutationFailure('missing', 'That task is no longer in the current plan'))
    }

    const task = (async (): Promise<TodoSnapshot> => {
      try {
        const raw = await deps.request<unknown>('todo.update_status', {
          actor: 'user',
          expected_revision: input.expectedRevision,
          item_id: input.itemId,
          session_id: input.sessionId,
          status: input.status
        })

        const response = parseTodoSnapshot(raw)

        if (!response || response.session_id !== input.sessionId) {
          throw new TodoMutationFailure('unavailable', 'Hermes returned an invalid task snapshot')
        }

        const current = deps.getSnapshot(input.sessionId)

        if (!current || response.generation >= current.generation) {
          deps.applySnapshot(response)

          return response
        }

        return current
      } catch (error) {
        const fromError = errorSnapshot(error, input.sessionId)

        if (fromError) {
          deps.applySnapshot(fromError)
        } else {
          try {
            const refreshed = parseTodoSnapshot(
              await deps.request<unknown>('todo.snapshot', { session_id: input.sessionId })
            )

            if (refreshed?.session_id === input.sessionId) {
              deps.applySnapshot(refreshed)
            } else {
              deps.applySnapshot(before)
            }
          } catch {
            const current = deps.getSnapshot(input.sessionId)

            if (!current || current.generation <= before.generation) {
              deps.applySnapshot(before)
            }
          }
        }

        if (error instanceof TodoMutationFailure) {
          throw error
        }

        throw new TodoMutationFailure(failureKind(error), errorMessage(error), { cause: error })
      } finally {
        pending.delete(key)
      }
    })()

    pending.set(key, task)

    return task
  }

  return {
    isPending: (sessionId: string, itemId: string) => pending.has(`${sessionId}\u0000${itemId}`),
    run
  }
}
