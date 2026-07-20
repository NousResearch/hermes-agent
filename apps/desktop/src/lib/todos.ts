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
  state: 'completed' | 'unfinished'
  timestamp?: number
  todos: TodoItem[]
}

export interface TodoHistoryMessage {
  id?: string
  parts?: unknown
  role?: string
  timestamp?: number
}

// Runtime status can legitimately advance between otherwise identical todo
// lists. History represents the task plan, so identity is id + content rather
// than a transient status value.
export const todoPlanSignature = (todos: readonly TodoItem[]) =>
  JSON.stringify(todos.map(({ content, id }) => ({ content, id })))

interface TodoHistoryCandidate {
  id: string
  todos: TodoItem[]
}

/** Full reconstruction used only where the caller has an authoritative
 * transcript replacement (resume/hydration/rewind). Newest snapshots come
 * first and repeated plans collapse to their newest occurrence. */
export function todoHistoryFromTranscript(messages: readonly TodoHistoryMessage[]): TodoHistorySnapshot[] {
  const seen = new Set<string>()
  const snapshots: TodoHistorySnapshot[] = []

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]

    if (!message || message.role !== 'assistant' || !Array.isArray(message.parts)) {
      continue
    }

    const candidates: TodoHistoryCandidate[] = []
    const messageSeen = new Set<string>()

    for (let partIndex = message.parts.length - 1; partIndex >= 0; partIndex -= 1) {
      const part = message.parts[partIndex]

      if (!isRecord(part) || part.type !== 'tool-call' || part.toolName !== 'todo') {
        continue
      }

      const todos = parseTodos(part.todos) ?? parseTodos(part.result) ?? parseTodos(part.args)

      if (!todos?.length) {
        continue
      }

      const signature = todoPlanSignature(todos)

      if (messageSeen.has(signature)) {
        continue
      }

      messageSeen.add(signature)
      const toolCallId = typeof part.toolCallId === 'string' && part.toolCallId ? part.toolCallId : `part-${partIndex}`
      candidates.push({ id: toolCallId, todos })
    }

    for (const candidate of candidates) {
      const signature = todoPlanSignature(candidate.todos)

      if (seen.has(signature)) {
        continue
      }

      seen.add(signature)
      const messageId = message.id || `assistant-turn-${index}`
      snapshots.push({
        id: candidates.length > 1 ? `${messageId}:${candidate.id}` : messageId,
        state: candidate.todos.some(todo => todo.status === 'pending' || todo.status === 'in_progress')
          ? 'unfinished'
          : 'completed',
        timestamp: message.timestamp,
        todos: [...candidate.todos]
      })
    }
  }

  return snapshots
}
