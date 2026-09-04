export type TodoStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled'

export interface TodoItem {
  content: string
  id: string
  /** Optional id of another item — renders this as a nested subtask. */
  parent?: string
  status: TodoStatus
}

/** One item from a `merge: true` write. Content is optional so a status-only
 *  patch still applies to an existing row. */
export interface TodoPatch {
  content?: string
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
    const parent = String(item.parent ?? '').trim()

    return id && content ? [{ content, id, status: item.status, ...(parent && parent !== id ? { parent } : {}) }] : []
  })
}

function parsePatchArray(value: unknown[]): TodoPatch[] {
  return value.flatMap(item => {
    if (!isRecord(item) || !isStatus(item.status)) {
      return []
    }

    const id = String(item.id ?? '').trim()

    if (!id) {
      return []
    }

    const content = String(item.content ?? '').trim()

    return content ? [{ content, id, status: item.status }] : [{ id, status: item.status }]
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

/** DFS order of a (possibly nested) todo list: [item, depth] pairs, parents
 *  before children. Dangling/cyclic parents degrade to depth 0. */
export function todoTree(todos: readonly TodoItem[]): [TodoItem, number][] {
  const ids = new Set(todos.map(t => t.id))
  const kids = new Map<string, TodoItem[]>()
  const roots: TodoItem[] = []

  for (const t of todos) {
    if (t.parent && ids.has(t.parent) && t.parent !== t.id) {
      const list = kids.get(t.parent) ?? []
      list.push(t)
      kids.set(t.parent, list)
    } else {
      roots.push(t)
    }
  }

  const out: [TodoItem, number][] = []
  const seen = new Set<string>()

  const walk = (item: TodoItem, depth: number) => {
    if (seen.has(item.id)) {
      return
    }

    seen.add(item.id)
    out.push([item, depth])

    for (const kid of kids.get(item.id) ?? []) {
      walk(kid, depth + 1)
    }
  }

  for (const root of roots) {
    walk(root, 0)
  }

  // Cycle members never reach a root — append them flat so nothing is lost.
  for (const t of todos) {
    if (!seen.has(t.id)) {
      seen.add(t.id)
      out.push([t, 0])
    }
  }

  return out
}

function parsePatch(value: unknown, depth: number): null | TodoPatch[] {
  if (depth > 2) {
    return null
  }

  if (Array.isArray(value)) {
    return parsePatchArray(value)
  }

  if (typeof value === 'string' && value.trim()) {
    try {
      return parsePatch(JSON.parse(value), depth + 1)
    } catch {
      return null
    }
  }

  if (isRecord(value) && Object.hasOwn(value, 'todos')) {
    return parsePatch(value.todos, depth + 1)
  }

  return null
}

export const parseTodoPatch = (value: unknown): null | TodoPatch[] => parsePatch(value, 0)

export const todoArgsWantMerge = (args: unknown): boolean => isRecord(args) && args.merge === true

/** Same as TodoStore.write(merge=True): update by id, append new items. */
export function mergeTodoItems(current: readonly TodoItem[], patch: readonly TodoPatch[]): TodoItem[] {
  const next = current.map(item => ({ ...item }))
  const indexById = new Map(next.map((item, index) => [item.id, index]))

  for (const item of patch) {
    const index = indexById.get(item.id)

    if (index === undefined) {
      next.push({ content: item.content?.trim() || '(no description)', id: item.id, status: item.status })
      indexById.set(item.id, next.length - 1)

      continue
    }

    if (item.content) {
      next[index].content = item.content
    }

    next[index].status = item.status
  }

  return next
}

/** Live tool event to the next list. `payload.todos` / `result` is the full
 *  store, so replace. `args` with `merge: true` patches by id so a status-only
 *  start event does not wipe the rest of the checklist. */
export function nextTodosFromToolEvent(
  current: readonly TodoItem[],
  payload: { args?: unknown; arguments?: unknown; result?: unknown; todos?: unknown }
): null | TodoItem[] {
  const fromResult = parseTodos(payload.todos) ?? parseTodos(payload.result)

  if (fromResult) {
    return fromResult
  }

  const args = payload.args ?? payload.arguments

  if (todoArgsWantMerge(args)) {
    const patch = parseTodoPatch(args)

    return patch && patch.length > 0 ? mergeTodoItems(current, patch) : null
  }

  return parseTodos(args)
}

function parseRevision(value: unknown, depth: number): null | number {
  if (depth > 2) {
    return null
  }

  if (typeof value === 'string' && value.trim()) {
    try {
      return parseRevision(JSON.parse(value), depth + 1)
    } catch {
      return null
    }
  }

  if (!isRecord(value)) {
    return null
  }

  if (typeof value.revision === 'number' && Number.isSafeInteger(value.revision) && value.revision >= 0) {
    return value.revision
  }

  return Object.hasOwn(value, 'result') ? parseRevision(value.result, depth + 1) : null
}

export const parseTodoRevision = (value: unknown): null | number => parseRevision(value, 0)

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
        // Always `messageId:toolCallId`. Suffixing only when a turn happened to
        // carry multiple plans made the same snapshot's id flip between
        // `messageId` and `messageId:toolCallId` across rebuilds (a later turn
        // could add or collapse candidates), needlessly remounting the row.
        id: `${messageId}:${candidate.id}`,
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
