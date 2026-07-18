export type TodoStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled'

export interface TodoItem {
  content: string
  id: string
  status: TodoStatus
}

const STATUSES: readonly TodoStatus[] = ['pending', 'in_progress', 'completed', 'cancelled']

const isRecord = (v: unknown): v is Record<string, unknown> => Boolean(v && typeof v === 'object' && !Array.isArray(v))
const isStatus = (v: unknown): v is TodoStatus => (STATUSES as readonly string[]).includes(v as string)

function parseArray(value: unknown[]): TodoItem[] {
  return value.flatMap(item => {
    if (!isRecord(item) || !isStatus(item.status)) {
      return []
    }

    const id = String(item.id ?? '').trim()
    const content = String(item.content ?? '').trim()

    return id && content ? [{ content, id, status: item.status }] : []
  })
}

function parse(value: unknown, depth: number): null | TodoItem[] {
  if (depth > 2) {
    return null
  }

  if (Array.isArray(value)) {
    return parseArray(value)
  }

  if (typeof value === 'string' && value.trim()) {
    try {
      return parse(JSON.parse(value), depth + 1)
    } catch {
      return null
    }
  }

  if (isRecord(value) && Object.hasOwn(value, 'todos')) {
    return parse(value.todos, depth + 1)
  }

  return null
}

export const parseTodos = (value: unknown): null | TodoItem[] => parse(value, 0)

/** Latest parseable todo list from one message's aui content parts (tool-call
 *  parts named `todo`; live parts carry `todos`, hydrated ones args/result). */
export function todosFromMessageContent(content: unknown): null | TodoItem[] {
  if (!Array.isArray(content)) {
    return null
  }

  let latest: null | TodoItem[] = null

  for (const part of content) {
    if (!isRecord(part) || part.type !== 'tool-call' || part.toolName !== 'todo') {
      continue
    }

    const parsed = parseTodos(part.todos) ?? parseTodos(part.result) ?? parseTodos(part.args)

    if (parsed !== null) {
      latest = parsed
    }
  }

  return latest
}

/** Current todo state for a whole transcript — the last list wins. */
export function latestSessionTodos(messages: readonly { parts?: unknown }[]): null | TodoItem[] {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const todos = todosFromMessageContent(messages[i]?.parts)

    if (todos !== null) {
      return todos
    }
  }

  return null
}

export interface TodoHistorySnapshot {
  id: string
  state: 'completed' | 'running' | 'unfinished'
  timestamp?: number
  todos: TodoItem[]
}

export interface SessionTodoHistory {
  current: TodoHistorySnapshot | null
  snapshots: TodoHistorySnapshot[]
}

export interface TodoHistoryMessage {
  id?: string
  pending?: boolean
  parts?: unknown
  role?: string
  timestamp?: number
}

const todosSignature = (todos: readonly TodoItem[]) =>
  JSON.stringify(todos.map(({ content, id, status }) => ({ content, id, status })))

/**
 * Preserve reference identity while streaming changes only non-todo text. The
 * composer can subscribe to this coarse derivation without re-rendering on
 * every token. Pending assistant turns are retained before their first todo
 * call so stale live-store linger cannot be promoted into the new turn.
 */
export function createTodoHistoryMessagesSelector() {
  let previousKey = ''
  let previous: readonly TodoHistoryMessage[] = []

  return (messages: readonly TodoHistoryMessage[]): readonly TodoHistoryMessage[] => {
    const relevant: TodoHistoryMessage[] = []
    const keyParts: string[] = []

    messages.forEach((message, index) => {
      if (message.role !== 'assistant') {
        return
      }

      const todos = todosFromMessageContent(message.parts)
      const pending = message.pending === true

      if (todos === null && !pending) {
        return
      }

      relevant.push(message)
      keyParts.push(
        `${message.id || index}:${message.timestamp ?? ''}:${pending ? 1 : 0}:${todos === null ? 'none' : todosSignature(todos)}`
      )
    })

    const key = keyParts.join('|')

    if (key === previousKey) {
      return previous
    }

    previousKey = key
    previous = relevant

    return previous
  }
}

/**
 * Reconstruct task-list history from the authoritative session transcript.
 * Each assistant message is one turn, so intermediate todo calls collapse to
 * that message's final snapshot. Identical lists retain only their newest
 * occurrence. The optional live store is presentation-only overlay data and is
 * never copied into durable state.
 */
export function sessionTodoHistory(
  messages: readonly TodoHistoryMessage[],
  liveTodos: readonly TodoItem[] | null = null,
  running = false
): SessionTodoHistory {
  const turnSnapshots = messages.flatMap((message, index) => {
    if (message.role !== 'assistant') {
      return []
    }

    const todos = todosFromMessageContent(message.parts)

    return todos !== null
      ? [
          {
            id: message.id || `assistant-turn-${index}`,
            pending: message.pending === true,
            timestamp: message.timestamp,
            todos
          }
        ]
      : []
  })

  const liveTranscriptSnapshot = running ? turnSnapshots.findLast(snapshot => snapshot.pending) : undefined

  const currentTodos = liveTranscriptSnapshot?.todos.length
    ? liveTodos === null
      ? liveTranscriptSnapshot.todos
      : liveTodos
    : undefined

  const current = currentTodos?.length
    ? {
        id: liveTranscriptSnapshot?.id || 'live',
        state: 'running' as const,
        timestamp: liveTranscriptSnapshot?.timestamp,
        todos: [...currentTodos]
      }
    : null

  const seen = new Set<string>()
  const snapshots: TodoHistorySnapshot[] = []

  for (let index = turnSnapshots.length - 1; index >= 0; index -= 1) {
    const snapshot = turnSnapshots[index]

    if (!snapshot || snapshot.todos.length === 0 || (running && snapshot === liveTranscriptSnapshot)) {
      continue
    }

    const signature = todosSignature(snapshot.todos)

    if (seen.has(signature)) {
      continue
    }

    seen.add(signature)
    snapshots.push({
      id: snapshot.id,
      state: snapshot.todos.some(todo => todo.status === 'pending' || todo.status === 'in_progress')
        ? 'unfinished'
        : 'completed',
      timestamp: snapshot.timestamp,
      todos: [...snapshot.todos]
    })
  }

  return { current, snapshots }
}
