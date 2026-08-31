export type TodoStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled'

export interface TodoItem {
  content: string
  id: string
  status: TodoStatus
}

/** Full authoritative snapshot from the gateway's `todo.snapshot` /
 *  `todo.update_status` RPCs and the `todo.updated` event. The revision +
 *  generation pair is what makes the list authoritative: revision is the
 *  expected-revision token for human Mark done/Reopen CAS writes, generation
 *  orders out-of-band updates (higher generation wins). A payload without
 *  both integers is display-only and never grants mutation authority. */
export interface TodoSnapshot {
  generation: number
  revision: number
  session_id: string
  todos: TodoItem[]
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

function parseAuthoritativeArray(value: unknown[]): TodoItem[] | null {
  const todos: TodoItem[] = []

  for (const item of value) {
    if (
      !isRecord(item) ||
      typeof item.id !== 'string' ||
      typeof item.content !== 'string' ||
      !isStatus(item.status)
    ) {
      return null
    }

    const id = item.id.trim()
    const content = item.content.trim()

    if (!id || !content) {
      return null
    }

    todos.push({ content, id, status: item.status })
  }

  return todos
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

const asInt = (value: unknown): null | number => {
  // Strict: the gateway's JSON-RPC layer always sends real numbers, and a
  // sloppy string coercion would let a malformed payload mint authority.
  // Booleans are numbers in JS — reject explicitly.
  if (typeof value === 'number' && Number.isInteger(value) && value >= 0) {
    return value
  }

  return null
}

/** Parse one authoritative full snapshot. Returns null for anything without
 *  BOTH integer version fields — that shape (a bare todo list) is deliberately
 *  accepted only by {@link parseTodos} as display-only and must never be
 *  promoted to mutation authority. */
export const parseTodoSnapshot = (value: unknown): null | TodoSnapshot => {
  let candidate: unknown = value

  if (typeof candidate === 'string' && candidate.trim()) {
    try {
      candidate = JSON.parse(candidate)
    } catch {
      return null
    }
  }

  // {todos: <snapshot>} wrapper (mirror of parseTodos' peek).
  if (
    candidate &&
    typeof candidate === 'object' &&
    !Array.isArray(candidate) &&
    Object.hasOwn(candidate as object, 'todos')
  ) {
    const inner = (candidate as Record<string, unknown>).todos

    if (inner && typeof inner === 'object' && !Array.isArray(inner) && Object.hasOwn(inner as object, 'todos')) {
      candidate = inner
    }
  }

  if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
    return null
  }

  const record = candidate as Record<string, unknown>
  const revision = asInt(record.revision)
  const generation = asInt(record.generation)
  const sessionId = String(record.session_id ?? '').trim()

  if (revision === null || generation === null || !sessionId || !Array.isArray(record.todos)) {
    return null
  }

  const todos = parseAuthoritativeArray(record.todos)

  if (!todos) {
    return null
  }

  return {
    generation,
    revision,
    session_id: sessionId,
    todos
  }
}

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
