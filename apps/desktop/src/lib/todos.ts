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
const TODO_SNAPSHOT_HEADER = '[Your active task list was preserved across context compression]'
const TODO_SNAPSHOT_FORMAT_V2 = '[Todo carrier format: 2]'
const PRUNED_SKILL_NOTICE_HEADER = '[Skills pruned during compression — reload before acting on these tasks]'
const PRUNED_SKILL_NOTICE_PREFIX =
  'The task list above crossed the compression boundary verbatim, but the skill instructions that governed it were pruned. Before executing any preserved task that depends on these skills, reload them first: '
const PRUNED_SKILL_NOTICE_SUFFIX =
  '. After reloading, re-check that each pending task is still justified — findings recorded before the boundary may have invalidated it.'
const TODO_ITEM_RE = /^( *)- \[([x> ~])\] (.+?)\. (.+) \((pending|in_progress|completed|cancelled)\)$/
const TODO_ITEM_V2_RE = /^( *)- \[([x> ~])\] (\{.+\})$/
const TODO_STATUS_BY_MARKER: Record<string, TodoStatus> = {
  ' ': 'pending',
  '>': 'in_progress',
  '~': 'cancelled',
  x: 'completed'
}
const SKILL_RELOAD_CALLS_RE = /^skill_view\(name='[^'\r\n]+'\)(?:; skill_view\(name='[^'\r\n]+'\))*$/

/** The task tool is `todo_list` on the wire since the core-tool rename; `todo`
 *  survives as the legacy alias in stored transcripts and older backends. */
export const isTodoToolName = (name: unknown): boolean => name === 'todo_list' || name === 'todo'

const isRecord = (v: unknown): v is Record<string, unknown> => Boolean(v && typeof v === 'object' && !Array.isArray(v))
const isStatus = (v: unknown): v is TodoStatus => (STATUSES as readonly string[]).includes(v as string)

function parseArray(value: unknown[]): null | TodoItem[] {
  const parsed: TodoItem[] = []

  for (const item of value) {
    if (!isRecord(item) || !isStatus(item.status)) {
      return null
    }

    if (typeof item.id !== 'string' || typeof item.content !== 'string') {
      return null
    }
    if (item.parent != null && typeof item.parent !== 'string') {
      return null
    }

    const id = item.id.trim()
    const content = item.content.trim()
    const parent = typeof item.parent === 'string' ? item.parent.trim() : ''

    if (!id || !content) {
      return null
    }

    parsed.push({ content, id, status: item.status, ...(parent && parent !== id ? { parent } : {}) })
  }

  return parsed
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

function parseMetadata(value: unknown): null | Record<string, unknown> {
  let parsed = value

  if (typeof parsed === 'string' && parsed.trim()) {
    try {
      parsed = JSON.parse(parsed)
    } catch {
      return null
    }
  }

  return isRecord(parsed) ? parsed : null
}

/** Whether display metadata explicitly identifies a todo compaction carrier. */
export function isTodoSnapshotMetadata(value: unknown): boolean {
  const snapshot = parseMetadata(value)?.todo_snapshot

  return snapshot === true || isRecord(snapshot)
}

/** Structured todo state carried by a compaction message's display metadata. */
export function todosFromSnapshotMetadata(value: unknown): null | TodoItem[] {
  const snapshot = parseMetadata(value)?.todo_snapshot

  return isRecord(snapshot) ? parseTodos(snapshot.todos) : null
}

function isProducerPrunedSkillNotice(lines: string[]): boolean {
  if (lines.length !== 3 || lines[0] !== '' || lines[1] !== PRUNED_SKILL_NOTICE_HEADER) {
    return false
  }

  const notice = lines[2]

  if (!notice.startsWith(PRUNED_SKILL_NOTICE_PREFIX) || !notice.endsWith(PRUNED_SKILL_NOTICE_SUFFIX)) {
    return false
  }

  const calls = notice.slice(PRUNED_SKILL_NOTICE_PREFIX.length, -PRUNED_SKILL_NOTICE_SUFFIX.length)

  return SKILL_RELOAD_CALLS_RE.test(calls)
}

/** Recover task state from an exact pre-metadata standalone compaction carrier.
 * Ambiguous composite or malformed prose fails closed and remains ordinary text. */
export function todosFromLegacySnapshotContent(value: unknown): null | TodoItem[] {
  if (typeof value !== 'string' || (value.includes('\r') && !value.includes('\r\n'))) {
    return null
  }

  const lines = value.replaceAll('\r\n', '\n').split('\n')

  if (lines[0] !== TODO_SNAPSHOT_HEADER) {
    return null
  }

  const todos: TodoItem[] = []
  const ancestors: TodoItem[] = []
  const ids = new Set<string>()
  const versioned = lines[1] === TODO_SNAPSHOT_FORMAT_V2
  let lineIndex = versioned ? 2 : 1

  for (; lineIndex < lines.length; lineIndex += 1) {
    let indent: string
    let marker: string
    let id: string
    let content: string
    let status: TodoStatus

    if (versioned) {
      const match = TODO_ITEM_V2_RE.exec(lines[lineIndex])

      if (!match) {
        break
      }

      let payload: unknown

      try {
        payload = JSON.parse(match[3])
      } catch {
        return null
      }

      if (
        !isRecord(payload) ||
        Object.keys(payload).length !== 3 ||
        typeof payload.id !== 'string' ||
        typeof payload.content !== 'string' ||
        !isStatus(payload.status)
      ) {
        return null
      }

      ;[, indent, marker] = match
      id = payload.id
      content = payload.content
      status = payload.status
    } else {
      const match = TODO_ITEM_RE.exec(lines[lineIndex])

      if (!match) {
        break
      }

      const statusValue = match[5]

      ;[, indent, marker, id, content] = match
      status = statusValue as TodoStatus

      // V1 has one unescaped `. ` separator. A second occurrence is
      // indistinguishable between an id suffix and a content prefix.
      if (content.includes('. ')) {
        return null
      }
    }

    if (
      indent.length % 2 !== 0 ||
      id !== id.trim() ||
      content !== content.trim() ||
      !id ||
      !content ||
      ids.has(id) ||
      TODO_STATUS_BY_MARKER[marker] !== status
    ) {
      return null
    }

    const depth = indent.length / 2

    if (depth > ancestors.length || (depth > 0 && !ancestors[depth - 1])) {
      return null
    }

    const parent = depth > 0 ? ancestors[depth - 1].id : undefined
    const todo = { content, id, ...(parent ? { parent } : {}), status }

    todos.push(todo)
    ids.add(id)
    ancestors.length = depth
    ancestors[depth] = todo
  }

  if (!todos.length) {
    return null
  }

  const remainder = lines.slice(lineIndex)

  if (remainder.length && !isProducerPrunedSkillNotice(remainder)) {
    return null
  }

  // format_for_injection omits completed/cancelled leaves. Such an item can
  // appear only as an ancestor of an active task, so reject hand-written lookalikes.
  const hasRetainedChild = new Set<string>()

  for (let index = todos.length - 1; index >= 0; index -= 1) {
    const todo = todos[index]
    const active = todo.status === 'pending' || todo.status === 'in_progress'

    if (!active && !hasRetainedChild.has(todo.id)) {
      return null
    }
    if (todo.parent) {
      hasRetainedChild.add(todo.parent)
    }
  }

  return todos
}

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
    if (!isRecord(part) || part.type !== 'tool-call' || !isTodoToolName(part.toolName)) {
      continue
    }

    const parsed = parseTodos(part.todos) ?? parseTodos(part.result) ?? parseTodos(part.args)

    if (parsed !== null) {
      latest = parsed
    }
  }

  return latest
}

export interface SessionTodoState {
  source: 'carrier' | 'tool'
  todos: TodoItem[]
}

type TodoHistoryMessage = {
  args?: unknown
  content?: unknown
  display_metadata?: unknown
  display_kind?: unknown
  name?: unknown
  parts?: unknown
  result?: unknown
  role?: unknown
  text?: unknown
  todo?: unknown
  todos?: unknown
  tool_call_id?: unknown
  tool_calls?: unknown
  tool_name?: unknown
}

function todoCall(call: unknown): null | { id: string; name: string; todos: null | TodoItem[] } {
  if (!isRecord(call)) {
    return null
  }

  const fn = isRecord(call.function) ? call.function : call
  const name = String(fn.name ?? call.name ?? '').trim()
  const id = String(call.id ?? call.tool_call_id ?? '').trim()
  const todos = name === 'todo' ? (parseTodos(fn.arguments) ?? parseTodos(call.args)) : null

  return { id, name, todos }
}

/** Current todo state plus provenance for raw REST/gateway rows or projected chat messages. */
export function latestSessionTodoState(messages: readonly TodoHistoryMessage[]): null | SessionTodoState {
  const pendingTodoCalls = new Map<string, unknown>()
  let latest: null | SessionTodoState = null

  for (const message of messages) {
    const partTodos = todosFromMessageContent(message.parts)

    if (partTodos !== null) {
      latest = { source: 'tool', todos: partTodos }
    }

    if (message.role === 'assistant' || message.role === 'user' || message.role === 'system') {
      pendingTodoCalls.clear()
      if (message.role === 'assistant' && Array.isArray(message.tool_calls)) {
        for (const rawCall of message.tool_calls) {
          const call = todoCall(rawCall)
          if (call?.id && call.name === 'todo') {
            pendingTodoCalls.set(call.id, rawCall)
          }
        }
      }
    }

    if (message.role === 'tool') {
      const callId = String(message.tool_call_id ?? '').trim()
      const matchingCall = pendingTodoCalls.get(callId)
      pendingTodoCalls.delete(callId)

      if (matchingCall !== undefined) {
        const todos =
          parseTodos(message.todos) ??
          parseTodos(message.result) ??
          parseTodos(message.content) ??
          parseTodos(message.text)

        if (todos !== null) {
          latest = { source: 'tool', todos }
        }
      }
    }

    const carrierTodos = todosFromSnapshotMetadata(message.display_metadata)

    if (carrierTodos !== null) {
      latest = { source: 'carrier', todos: carrierTodos }
      continue
    }

    const legacyCarrierTodos =
      message.role === 'user' && message.display_kind == null && message.display_metadata == null
        ? todosFromLegacySnapshotContent(message.content ?? message.text)
        : null

    if (legacyCarrierTodos !== null) {
      latest = { source: 'carrier', todos: legacyCarrierTodos }
    }
  }

  return latest
}

/** Current todo state for a whole transcript — the last list wins. */
export function latestSessionTodos(messages: readonly TodoHistoryMessage[]): null | TodoItem[] {
  return latestSessionTodoState(messages)?.todos ?? null
}
