import type { ThreadMessageLike } from '@assistant-ui/react'

import { dedupeGeneratedImageEchoesInParts } from '@/lib/generated-images'
import { mediaDisplayLabel, mediaMarkdownHref } from '@/lib/media'
import { normalize } from '@/lib/text'
import { parseTodos } from '@/lib/todos'
import type { SessionMessage, UsageStats } from '@/types/hermes'

export type ChatMessagePart = Exclude<ThreadMessageLike['content'], string>[number]

export type ChatMessage = {
  id: string
  role: SessionMessage['role']
  parts: ChatMessagePart[]
  timestamp?: number
  pending?: boolean
  error?: string
  branchGroupId?: string
  hidden?: boolean
  /** Composer attachment ref strings (`@file:...`, `@image:...`) sent with this user message. */
  attachmentRefs?: string[]
}

export function isSessionBusyMessage(value: unknown): boolean {
  return /(?:^|\b)session busy(?:\b|$)/i.test(typeof value === 'string' ? value : String(value ?? ''))
}

export type GatewayEventPayload = {
  text?: string
  rendered?: string
  status?: string
  compression_exhausted?: boolean
  message?: string
  id?: string
  name?: string
  tool_id?: string
  tool_call_id?: string
  args?: unknown
  arguments?: unknown
  context?: string
  input?: unknown
  preview?: string
  result?: unknown
  summary?: string
  error?: string | boolean
  inline_diff?: string
  duration_s?: number
  todos?: unknown
  model?: string
  provider?: string
  reasoning_effort?: string
  service_tier?: string
  fast?: boolean
  yolo?: boolean
  running?: boolean
  cwd?: string
  branch?: string
  credential_warning?: string
  personality?: string
  usage?: Partial<UsageStats>
  // agent.terminal.output — live chunk for a read-only agent terminal tab
  process_id?: string
  chunk?: string
  // clarify.request
  request_id?: string
  question?: string
  choices?: string[] | null
  // approval.request (dangerous command / execute_code) — session-keyed
  command?: string
  description?: string
  // False when a tirith content-security warning forbids a permanent allow.
  allow_permanent?: boolean
  // secret.request (skill credential capture)
  env_var?: string
  prompt?: string
  // terminal.read.request (GUI agent reading the in-app terminal pane)
  start?: number
  count?: number
  // status.update (kind=process → background process completion/watch-match)
  kind?: string
  // session.title (live auto-title push) — stored session id + generated title
  session_id?: string
  title?: string
  // moa.reference / moa.aggregating (Mixture of Agents per-model relay)
  label?: string
  index?: number
  aggregator?: string
}

export function textPart(text: string): ChatMessagePart {
  return { type: 'text', text }
}

export function reasoningPart(text: string): ChatMessagePart {
  return { type: 'reasoning', text }
}

const MEDIA_LINE_RE = /(^|\n)[\t ]*[`"']?MEDIA:\s*(?<line>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)[`"']?[\t ]*(\n|$)/g

const MEDIA_TAG_RE = /[`"']?MEDIA:\s*(?<inline>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)[`"']?/g

function unquoteMediaPath(value: string): string {
  const trimmed = value.trim()
  const quote = trimmed[0]

  return quote && quote === trimmed.at(-1) && ['"', "'", '`'].includes(quote) ? trimmed.slice(1, -1) : trimmed
}

function mediaLink(value: string): string {
  const path = unquoteMediaPath(value)

  return `[${mediaDisplayLabel(path)}](${mediaMarkdownHref(path)})`
}

export function renderMediaTags(text: string): string {
  return text
    .replace(
      MEDIA_LINE_RE,
      (_match, lead: string, value: string, trailer: string) => `${lead}${mediaLink(value)}${trailer}`
    )
    .replace(MEDIA_TAG_RE, (_match, value: string) => mediaLink(value))
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
}

export function assistantTextPart(text: string): ChatMessagePart {
  return textPart(renderMediaTags(text))
}

export function chatMessageText(message: ChatMessage): string {
  return message.parts
    .filter((part): part is Extract<ChatMessagePart, { type: 'text' }> => part.type === 'text')
    .map(part => part.text)
    .join('')
}

const ATTACHED_CONTEXT_MARKER_RE = /(?:^|\n)--- Attached Context ---\s*\n/
const CONTEXT_WARNINGS_MARKER_RE = /(?:^|\n)--- Context Warnings ---[\s\S]*$/
const CONTEXT_REF_RE = /@(file|folder|url|image|tool|terminal):(?:"[^"\n]+"|'[^'\n]+'|`[^`\n]+`|\S+)/g

function textFromUnknown(value: unknown, depth = 0): string {
  if (typeof value === 'string') {
    return value
  }

  if (value === null || value === undefined) {
    return ''
  }

  if (depth > 2) {
    return ''
  }

  if (Array.isArray(value)) {
    return value.map(item => textFromUnknown(item, depth + 1)).join('')
  }

  if (typeof value === 'object') {
    const row = value as Record<string, unknown>
    const textValue = row.text ?? row.output_text ?? row.content ?? row.message
    const nestedText = textFromUnknown(textValue, depth + 1)

    if (nestedText) {
      return nestedText
    }

    try {
      return JSON.stringify(value)
    } catch {
      return ''
    }
  }

  return String(value)
}

function displayContentForMessage(role: SessionMessage['role'], content: unknown): string {
  const textContent = textFromUnknown(content)

  if (role !== 'user') {
    return textContent
  }

  const marker = textContent.match(ATTACHED_CONTEXT_MARKER_RE)

  if (!marker || marker.index === undefined) {
    return textContent.replace(CONTEXT_WARNINGS_MARKER_RE, '').trim()
  }

  const visibleText = textContent.slice(0, marker.index).replace(CONTEXT_WARNINGS_MARKER_RE, '').trim()
  const attachedContext = textContent.slice(marker.index + marker[0].length)
  const refs = [...new Set(Array.from(attachedContext.matchAll(CONTEXT_REF_RE)).map(match => match[0]))]

  return [refs.join('\n'), visibleText].filter(Boolean).join('\n\n') || visibleText
}

const STREAM_PART: Record<'reasoning' | 'text', (text: string) => ChatMessagePart> = {
  reasoning: reasoningPart,
  text: textPart
}

// Coalesce a streaming delta into the most recent same-type part within the
// current segment, where a segment is bounded by any non-streaming part (a
// tool call, image, …). The opposite streaming channel (text <-> reasoning) is
// transparent, so a reasoning burst between two content deltas can't shred one
// sentence into text / Thinking / text — the fragmentation models that
// interleave reasoning_content + content otherwise produce. Tool calls still
// open a fresh part, preserving narration order across steps.
function appendStreamPart(
  parts: ChatMessagePart[],
  type: 'reasoning' | 'text',
  delta: string
): { index: number; parts: ChatMessagePart[] } {
  const next = [...parts]

  for (let i = next.length - 1; i >= 0; i--) {
    const part = next[i]

    if (part.type === type) {
      next[i] = { ...part, text: `${(part as { text: string }).text}${delta}` } as ChatMessagePart

      return { index: i, parts: next }
    }

    if (part.type !== 'text' && part.type !== 'reasoning') {
      break
    }
  }

  next.push(STREAM_PART[type](delta))

  return { index: next.length - 1, parts: next }
}

export function appendTextPart(parts: ChatMessagePart[], delta: string): ChatMessagePart[] {
  return appendStreamPart(parts, 'text', delta).parts
}

export function appendReasoningPart(parts: ChatMessagePart[], delta: string): ChatMessagePart[] {
  return appendStreamPart(parts, 'reasoning', delta).parts
}

export function appendAssistantTextPart(parts: ChatMessagePart[], delta: string): ChatMessagePart[] {
  const { index, parts: next } = appendStreamPart(parts, 'text', delta)
  const part = next[index]

  if (part?.type !== 'text') {
    return next
  }

  const mayContainMedia =
    delta.includes('MEDIA:') || delta.includes('DIA:') || delta.includes('EDIA:') || delta.includes('IA:')

  if (mayContainMedia || part.text.includes('MEDIA:')) {
    const rendered = renderMediaTags(part.text)

    if (rendered !== part.text) {
      next[index] = { ...part, text: rendered }
    }
  }

  return next
}

export function hasToolPart(message: ChatMessage): boolean {
  return message.parts.some(part => part.type === 'tool-call')
}

function toolId(payload: GatewayEventPayload | undefined): string {
  return payload?.tool_id || payload?.tool_call_id || payload?.id || ''
}

let liveToolCounter = 0

function nextLiveToolId(name: string): string {
  liveToolCounter += 1

  return `live-tool:${name}:${liveToolCounter}`
}

function firstStringField(record: Record<string, unknown>, keys: readonly string[]): string {
  for (const key of keys) {
    const value = record[key]

    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }

  return ''
}

function normalizeToolMatchValue(value: string): string {
  return normalize(value)
}

function collectToolMatchValues(query: string, context: string, preview: string): string[] {
  return [...new Set([query, context, preview].map(normalizeToolMatchValue).filter(Boolean))]
}

function toolPayloadMatchValues(payload: GatewayEventPayload | undefined): string[] {
  const payloadArgs = liveToolArgs(payload)
  const query = firstStringField(payloadArgs, ['search_term', 'query'])
  const context = typeof payload?.context === 'string' ? payload.context.trim() : ''
  const preview = typeof payload?.preview === 'string' ? payload.preview.trim() : ''

  return collectToolMatchValues(query, context, preview)
}

function toolPartMatchValues(part: ChatMessagePart): string[] {
  if (part.type !== 'tool-call' || !part.args || typeof part.args !== 'object') {
    return []
  }

  const args = part.args as Record<string, unknown>
  const query = firstStringField(args, ['search_term', 'query'])
  const context = typeof args.context === 'string' ? args.context.trim() : ''
  const preview = typeof args.preview === 'string' ? args.preview.trim() : ''

  return collectToolMatchValues(query, context, preview)
}

function hasToolMatchOverlap(left: string[], right: string[]): boolean {
  if (!left.length || !right.length) {
    return false
  }

  const rightSet = new Set(right)

  return left.some(value => rightSet.has(value))
}

function findToolPartIndex(
  parts: ChatMessagePart[],
  name: string,
  stableId: string,
  payload: GatewayEventPayload | undefined,
  phase: 'running' | 'complete'
): number {
  const matchValues = toolPayloadMatchValues(payload)
  const overlaps = (index: number) => hasToolMatchOverlap(matchValues, toolPartMatchValues(parts[index]))

  if (stableId) {
    const stableIndex = parts.findIndex(part => part.type === 'tool-call' && part.toolCallId === stableId)

    if (stableIndex >= 0) {
      return stableIndex
    }

    // Some live streams start without an id, then complete with one. Fall
    // through to pending same-name/context matching so the completion updates
    // the synthetic live row instead of appending a duplicate completed row.
    if (phase === 'running' && !matchValues.length) {
      return -1
    }
  }

  const pendingIndices = parts
    .map((part, index) => ({ part, index }))
    .filter(({ part }) => part.type === 'tool-call' && part.toolName === name && part.result === undefined)
    .map(({ index }) => index)

  if (pendingIndices.length === 0) {
    return -1
  }

  if (matchValues.length) {
    const contextualIndex = pendingIndices.find(overlaps)

    if (contextualIndex !== undefined) {
      return contextualIndex
    }
  }

  if (pendingIndices.length === 1) {
    const [singlePendingIndex] = pendingIndices

    if (phase === 'running' && matchValues.length && !overlaps(singlePendingIndex)) {
      return stableId ? singlePendingIndex : -1
    }

    return singlePendingIndex
  }

  // Completion events without stable IDs frequently arrive after multiple
  // same-name starts (parallel tool calls). Resolve them oldest-first so we
  // don't collapse an entire burst into a single row.
  if (phase === 'complete') {
    return pendingIndices[0]
  }

  if (stableId) {
    return pendingIndices[0]
  }

  // For progress/running events with no stable id, update the most-recent
  // pending same-name tool instead of creating a phantom extra row.
  return pendingIndices.at(-1) ?? -1
}

// Carry todo state across sparse progress payloads: if this todo event lacks
// a `todos` field, fall back to whatever we previously stored on the part.
function carryTodos(payload: GatewayEventPayload | undefined, ...prev: unknown[]): { todos: unknown } | undefined {
  if (payload && Object.hasOwn(payload, 'todos')) {
    const next = parseTodos(payload.todos)

    return next === null ? undefined : { todos: next }
  }

  if (payload?.name !== 'todo') {
    return undefined
  }

  for (const p of prev) {
    const carried = parseTodos(recordFromUnknown(p)?.todos)

    if (carried !== null) {
      return { todos: carried }
    }
  }

  return undefined
}

function toolArgs(payload: GatewayEventPayload | undefined, prevArgs?: unknown): Record<string, unknown> {
  const prev = parseMaybeJsonObject(prevArgs)
  const eventArgs = liveToolArgs(payload)

  return {
    ...prev,
    ...eventArgs,
    ...(payload?.context ? { context: payload.context } : {}),
    ...(payload?.preview ? { preview: payload.preview } : {}),
    ...carryTodos(payload, prevArgs)
  }
}

function toolResult(
  payload: GatewayEventPayload | undefined,
  prevResult?: unknown,
  prevArgs?: unknown
): Record<string, unknown> {
  const parsedResult = parseMaybeJsonObject(payload?.result)

  return {
    ...parsedResult,
    ...(payload?.inline_diff ? { inline_diff: payload.inline_diff } : {}),
    ...(payload?.summary ? { summary: payload.summary } : {}),
    ...(payload?.message ? { message: payload.message } : {}),
    ...(payload?.preview ? { preview: payload.preview } : {}),
    ...(payload?.duration_s !== undefined ? { duration_s: payload.duration_s } : {}),
    ...carryTodos(payload, prevResult, prevArgs),
    ...(payload?.error ? { error: payload.error } : {})
  }
}

export function upsertToolPart(
  parts: ChatMessagePart[],
  payload: GatewayEventPayload | undefined,
  phase: 'running' | 'complete'
): ChatMessagePart[] {
  const stableId = toolId(payload)
  const name = payload?.name || 'tool'
  const next = [...parts]

  const index = findToolPartIndex(next, name, stableId, payload, phase)

  const prev = index >= 0 ? next[index] : null
  const prevArgs = prev && 'args' in prev ? prev.args : undefined
  const prevResult = prev && 'result' in prev ? prev.result : undefined
  const args = toolArgs(payload, prevArgs)

  const id =
    stableId ||
    (prev && 'toolCallId' in prev && typeof prev.toolCallId === 'string' ? prev.toolCallId : '') ||
    nextLiveToolId(name)

  const base = {
    type: 'tool-call' as const,
    toolCallId: id,
    toolName: name,
    args: args as never,
    argsText: JSON.stringify(args),
    ...(phase === 'complete' && { result: toolResult(payload, prevResult, prevArgs), isError: Boolean(payload?.error) })
  } satisfies ChatMessagePart

  if (index === -1) {
    return [...next, base]
  }

  next[index] = { ...next[index], ...base }

  return next
}

function recordFromUnknown(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : null
}

function parseMaybeJsonObject(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }

  if (typeof value !== 'string' || !value.trim()) {
    return {}
  }

  try {
    const parsed = JSON.parse(value)

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

function firstNonEmptyObject(...values: unknown[]): Record<string, unknown> {
  for (const value of values) {
    const parsed = parseMaybeJsonObject(value)

    if (Object.keys(parsed).length > 0) {
      return parsed
    }
  }

  return {}
}

function liveToolArgs(payload: GatewayEventPayload | undefined): Record<string, unknown> {
  const direct = firstNonEmptyObject(payload?.args, payload?.arguments)
  const input = firstNonEmptyObject(payload?.input)
  const fn = recordFromUnknown(input.function)

  const nested = firstNonEmptyObject(
    input.args,
    input.arguments,
    input.parameters,
    input.input,
    fn?.arguments,
    fn?.args,
    fn?.parameters
  )

  return {
    ...input,
    ...nested,
    ...direct
  }
}

function parseStoredToolResult(content: unknown): unknown {
  if (content && typeof content === 'object') {
    return content
  }

  const textContent = textFromUnknown(content)

  if (!textContent.trim()) {
    return ''
  }

  try {
    return JSON.parse(textContent)
  } catch {
    return textContent
  }
}

function toolPartFromStoredCall(call: unknown, fallbackIndex: number): ChatMessagePart {
  const row = recordFromUnknown(call) ?? {}
  const fn = recordFromUnknown(row.function)
  const id = String(row.id || row.tool_call_id || `stored-tool-${fallbackIndex}`)

  const toolName = String(
    row.name || row.tool_name || fn?.name || (recordFromUnknown(row.input)?.name as string | undefined) || 'tool'
  )

  const args = firstNonEmptyObject(fn?.arguments, row.arguments, row.args, row.input)

  return {
    type: 'tool-call',
    toolCallId: id,
    toolName,
    args: args as never,
    argsText: Object.keys(args).length ? JSON.stringify(args) : ''
  }
}

function applyStoredToolResult(messages: ChatMessage[], toolMessage: SessionMessage): boolean {
  const toolCallId = toolMessage.tool_call_id || undefined
  const toolName = toolMessage.tool_name || toolMessage.name || 'tool'
  const content = toolMessage.content || toolMessage.text || toolMessage.context || toolMessage.name

  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i]

    if (message.role !== 'assistant') {
      continue
    }

    const partIndex = message.parts.findIndex(
      part =>
        part.type === 'tool-call' &&
        ((toolCallId && part.toolCallId === toolCallId) || (!toolCallId && part.toolName === toolName))
    )

    if (partIndex < 0) {
      continue
    }

    const parts = [...message.parts]
    const existing = parts[partIndex]
    parts[partIndex] = {
      ...existing,
      result: parseStoredToolResult(content),
      isError: false
    } as ChatMessagePart
    messages[i] = { ...message, parts }

    return true
  }

  return false
}

function applyStoredToolResultToParts(parts: ChatMessagePart[], toolMessage: SessionMessage): ChatMessagePart[] | null {
  const toolCallId = toolMessage.tool_call_id || undefined
  const toolName = toolMessage.tool_name || toolMessage.name || 'tool'
  const content = toolMessage.content || toolMessage.text || toolMessage.context || toolMessage.name

  const partIndex = parts.findIndex(
    part =>
      part.type === 'tool-call' &&
      ((toolCallId && part.toolCallId === toolCallId) || (!toolCallId && part.toolName === toolName))
  )

  if (partIndex < 0) {
    return null
  }

  const next = [...parts]
  const existing = next[partIndex]
  next[partIndex] = {
    ...existing,
    result: parseStoredToolResult(content),
    isError: false
  } as ChatMessagePart

  return next
}

function storedToolMessagePart(toolMessage: SessionMessage, fallbackIndex: number): ChatMessagePart {
  const name = toolMessage.tool_name || toolMessage.name || 'tool'
  const context = textFromUnknown(toolMessage.context || toolMessage.text || toolMessage.content || '')
  const args = context ? { context } : {}

  return {
    type: 'tool-call',
    toolCallId: toolMessage.tool_call_id || `stored-tool-message-${fallbackIndex}`,
    toolName: name,
    args: args as never,
    argsText: Object.keys(args).length ? JSON.stringify(args) : '',
    result: context ? { context } : {},
    isError: false
  }
}

function withUniqueToolCallIds(messages: ChatMessage[]): ChatMessage[] {
  const seen = new Set<string>()

  return messages.map(message => {
    let changed = false

    const parts = message.parts.map((part, index) => {
      if (part.type !== 'tool-call') {
        return part
      }

      const id = part.toolCallId || `${message.id}-tool-${index}`

      if (!seen.has(id)) {
        seen.add(id)

        if (part.toolCallId) {
          return part
        }

        changed = true

        return { ...part, toolCallId: id } as ChatMessagePart
      }

      changed = true
      const uniqueId = `${id}-${message.id}-${index}`
      seen.add(uniqueId)

      return { ...part, toolCallId: uniqueId } as ChatMessagePart
    })

    return changed ? { ...message, parts } : message
  })
}

export function toChatMessages(messages: SessionMessage[]): ChatMessage[] {
  const result: ChatMessage[] = []
  let pendingToolParts: ChatMessagePart[] = []
  let pendingToolTimestamp: number | undefined
  let activeAssistantIndex: null | number = null

  const clearPendingTools = () => {
    pendingToolParts = []
    pendingToolTimestamp = undefined
  }

  const appendPartsToActiveAssistant = (parts: ChatMessagePart[], timestamp?: number): boolean => {
    if (activeAssistantIndex === null) {
      return false
    }

    const active = result[activeAssistantIndex]

    if (!active || active.role !== 'assistant') {
      activeAssistantIndex = null

      return false
    }

    active.parts = [...active.parts, ...parts]
    active.timestamp = timestamp ?? active.timestamp

    return true
  }

  const flushPendingTools = (index: number) => {
    if (!pendingToolParts.length) {
      return
    }

    if (!appendPartsToActiveAssistant(pendingToolParts, pendingToolTimestamp)) {
      result.push({
        id: `${pendingToolTimestamp || Date.now()}-${index}-tools`,
        role: 'assistant',
        parts: pendingToolParts,
        timestamp: pendingToolTimestamp
      })
      activeAssistantIndex = result.length - 1
    }

    clearPendingTools()
  }

  messages.forEach((message, index) => {
    if (message.role === 'tool') {
      const updatedPendingToolParts = applyStoredToolResultToParts(pendingToolParts, message)

      if (updatedPendingToolParts) {
        pendingToolParts = updatedPendingToolParts

        return
      }

      if (applyStoredToolResult(result, message)) {
        return
      }

      pendingToolParts = [...pendingToolParts, storedToolMessagePart(message, index)]
      pendingToolTimestamp ??= message.timestamp

      return
    }

    const content = message.content || message.text || message.context || message.name
    const displayContent = displayContentForMessage(message.role, content)
    const parts: ChatMessagePart[] = []

    const reasoning =
      message.reasoning ||
      message.reasoning_content ||
      (typeof message.reasoning_details === 'string' ? message.reasoning_details : '')

    if (reasoning && message.role === 'assistant') {
      parts.push(reasoningPart(reasoning))
    }

    if (displayContent) {
      parts.push(message.role === 'assistant' ? assistantTextPart(displayContent) : textPart(displayContent))
    }

    if (message.role === 'assistant' && Array.isArray(message.tool_calls)) {
      parts.push(...message.tool_calls.map((call, callIndex) => toolPartFromStoredCall(call, callIndex)))
    }

    if (!parts.length) {
      if (message.role !== 'assistant') {
        flushPendingTools(index)
        activeAssistantIndex = null
      }

      return
    }

    const isToolOnlyAssistant =
      message.role === 'assistant' && parts.length > 0 && parts.every(part => part.type === 'tool-call')

    if (isToolOnlyAssistant) {
      pendingToolParts = [...pendingToolParts, ...parts]
      pendingToolTimestamp ??= message.timestamp

      return
    }

    if (message.role === 'assistant') {
      if (pendingToolParts.length) {
        if (!appendPartsToActiveAssistant(pendingToolParts, message.timestamp ?? pendingToolTimestamp)) {
          parts.unshift(...pendingToolParts)
        }

        clearPendingTools()
      }

      const activeAssistant =
        activeAssistantIndex !== null && result[activeAssistantIndex]?.role === 'assistant'
          ? result[activeAssistantIndex]
          : null

      const currentHasToolCall = parts.some(part => part.type === 'tool-call')
      const activeHasToolCall = Boolean(activeAssistant?.parts.some(part => part.type === 'tool-call'))

      if (activeAssistant && (currentHasToolCall || activeHasToolCall)) {
        activeAssistant.parts = [...activeAssistant.parts, ...parts]
        activeAssistant.timestamp = message.timestamp ?? activeAssistant.timestamp

        return
      }
    } else {
      flushPendingTools(index)
    }

    result.push({
      id: `${message.timestamp || Date.now()}-${index}-${message.role}`,
      role: message.role,
      parts,
      timestamp: message.timestamp
    })

    activeAssistantIndex = message.role === 'assistant' ? result.length - 1 : null
  })
  flushPendingTools(messages.length)

  const withoutGeneratedImageEchoes = result.map(message =>
    message.role === 'assistant' ? { ...message, parts: dedupeGeneratedImageEchoesInParts(message.parts) } : message
  )

  return withUniqueToolCallIds(
    withoutGeneratedImageEchoes.filter(
      m =>
        !(m.role === 'assistant' && isSessionBusyMessage(m.error || chatMessageText(m))) &&
        (chatMessageText(m).trim() || m.parts.some(part => part.type !== 'text'))
    )
  )
}

export function preserveLocalAssistantErrors(
  nextMessages: ChatMessage[],
  currentMessages: ChatMessage[]
): ChatMessage[] {
  const displayableNextMessages = nextMessages.filter(message => !isSessionBusyMessage(message.error))
  const displayableCurrentMessages = currentMessages.filter(message => !isSessionBusyMessage(message.error))
  const localById = new Map(displayableCurrentMessages.map(message => [message.id, message]))

  const mergedNextMessages = displayableNextMessages.map(message => {
    if (message.role !== 'assistant' || message.error || message.hidden) {
      return message
    }

    const local = localById.get(message.id)

    if (!local || local.role !== 'assistant' || !local.error || local.hidden) {
      return message
    }

    return {
      ...message,
      error: local.error,
      pending: false
    }
  })

  const existingIds = new Set(mergedNextMessages.map(message => message.id))
  const preserveIds = new Set<string>()
  const normalize = (value: string) => value.replace(/\s+/g, ' ').trim()
  const tailUserInNext = [...mergedNextMessages].reverse().find(message => message.role === 'user' && !message.hidden)
  const tailUserText = tailUserInNext ? normalize(chatMessageText(tailUserInNext)) : ''
  const tailUserRefs = tailUserInNext ? (tailUserInNext.attachmentRefs ?? []).join('\n') : ''

  const matchesTailUserInNext = (candidate: ChatMessage) =>
    Boolean(tailUserInNext) &&
    normalize(chatMessageText(candidate)) === tailUserText &&
    (candidate.attachmentRefs ?? []).join('\n') === tailUserRefs

  for (let index = 0; index < displayableCurrentMessages.length; index += 1) {
    const message = displayableCurrentMessages[index]

    if (message.role !== 'assistant' || !message.error || message.hidden || existingIds.has(message.id)) {
      continue
    }

    preserveIds.add(message.id)

    for (let probe = index - 1; probe >= 0; probe -= 1) {
      const candidate = displayableCurrentMessages[probe]

      if (candidate.hidden) {
        continue
      }

      if (candidate.role === 'user' && !existingIds.has(candidate.id) && !matchesTailUserInNext(candidate)) {
        preserveIds.add(candidate.id)
      }

      break
    }
  }

  if (preserveIds.size === 0) {
    return mergedNextMessages
  }

  const preserved = displayableCurrentMessages
    .filter(message => preserveIds.has(message.id))
    .map(message => ({ ...message, pending: false }))

  return [...mergedNextMessages, ...preserved]
}

export function branchGroupForUser(userMessage: ChatMessage): string {
  return `branch:${userMessage.id}`
}
