import { firstStringField, normalize } from '@/lib/text'
import { parseTodos } from '@/lib/todos'
import type { SessionMessage } from '@/types/hermes'

import type { ChatMessage, ChatMessagePart, GatewayEventPayload } from './types'

function toolId(payload: GatewayEventPayload | undefined): string {
  return payload?.tool_id || payload?.tool_call_id || payload?.id || ''
}

let liveToolCounter = 0

function nextLiveToolId(name: string): string {
  liveToolCounter += 1

  return `live-tool:${name}:${liveToolCounter}`
}

function normalizeToolMatchValue(value: string): string {
  return normalize(value)
}

function collectToolMatchValues(query: string, context: string, preview: string): string[] {
  return [...new Set([query, context, preview].map(normalizeToolMatchValue).filter(Boolean))]
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

function toolPayloadMatchValues(payload: GatewayEventPayload | undefined): string[] {
  const payloadArgs = liveToolArgs(payload)

  // `question` is clarify's identifying arg: a synthetic row hydrated from
  // `clarify.request` (a fresh request id) must correlate with the `tool.start`
  // row (the model's tool_call_id) so the two ids don't produce a duplicate
  // clarify card — same correlation ClarifyToolPending uses for request↔args.
  // `server` is setup_mcp's identifying arg, for the identical reason.
  const query =
    firstStringField(payloadArgs, ['search_term', 'query', 'question', 'server', 'command', 'code', 'path']) ||
    batchClarifyMatchValue(payloadArgs.questions)

  const context = typeof payload?.context === 'string' ? payload.context.trim() : ''
  const preview = typeof payload?.preview === 'string' ? payload.preview.trim() : ''

  return collectToolMatchValues(query, context, preview)
}

/**
 * The batch-clarify counterpart of the `question` correlation key: a batch
 * payload has no top-level `question`, only `questions[]`, so without this
 * the request row and the tool.start row never match and the card mounts
 * twice. The joined per-question texts identify the batch the same way one
 * question text identifies a single prompt. The `\u0000` separator cannot
 * appear in real question text, so a batch key can never collide with a
 * single-question key.
 */
function batchClarifyMatchValue(questions: unknown): string {
  if (!Array.isArray(questions)) {
    return ''
  }

  const texts = questions
    .map(entry => {
      if (!entry || typeof entry !== 'object') {
        return ''
      }

      const question = (entry as Record<string, unknown>).question

      return typeof question === 'string' ? question.trim() : ''
    })
    .filter(Boolean)

  return texts.length > 0 ? texts.join('\u0000') : ''
}

function toolPartMatchValues(part: ChatMessagePart): string[] {
  if (part.type !== 'tool-call' || !part.args || typeof part.args !== 'object') {
    return []
  }

  const args = part.args as Record<string, unknown>

  const query =
    firstStringField(args, ['search_term', 'query', 'question', 'server', 'command', 'code', 'path']) ||
    batchClarifyMatchValue(args.questions)

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

function completeOpenStreamParts(parts: ChatMessagePart[], completedAt: number): ChatMessagePart[] {
  return parts.map(part =>
    (part.type === 'text' || part.type === 'reasoning') && part.completedAt === undefined
      ? ({ ...part, completedAt } as ChatMessagePart)
      : part
  )
}

export function upsertToolPart(
  parts: ChatMessagePart[],
  payload: GatewayEventPayload | undefined,
  phase: 'running' | 'complete',
  occurredAt = Date.now() / 1000
): ChatMessagePart[] {
  const stableId = toolId(payload)
  const name = payload?.name || 'tool'
  // A completion can be the first tool event observed after reconnect, so it
  // also constitutes a text/reasoning -> tool boundary when no start arrived.
  const next = completeOpenStreamParts(parts, occurredAt)

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
    timestamp: prev?.timestamp ?? occurredAt,
    ...(phase === 'complete' && {
      completedAt: occurredAt,
      result: toolResult(payload, prevResult, prevArgs),
      isError: Boolean(payload?.error)
    })
  } satisfies ChatMessagePart

  if (index === -1) {
    return [...next, base]
  }

  next[index] = { ...next[index], ...base }

  return next
}

/**
 * Turn-settle reconciliation: close every tool-call part that never received
 * its completion event. A `tool.complete` lost to a degraded websocket
 * (reconnect, profile swap, hidden window) leaves the part without a `result`,
 * which renders as a permanently spinning tool row even though the turn itself
 * completed. A settled session cannot have tools still running, so an open
 * part at settle time is a lost event, not live work. Pending messages are
 * left alone, and no-op calls return the input array unchanged.
 */
export function sealOpenToolParts(messages: ChatMessage[]): ChatMessage[] {
  let changed = false

  const next = messages.map(message => {
    if (message.role !== 'assistant' || message.pending) {
      return message
    }

    let partChanged = false

    const parts = message.parts.map(part => {
      if (part.type !== 'tool-call' || Object.hasOwn(part, 'result')) {
        return part
      }

      partChanged = true

      return { ...part, result: {} }
    })

    if (!partChanged) {
      return message
    }

    changed = true

    return { ...message, parts }
  })

  return changed ? next : messages
}

// ── Stored-tool conversion (hydration path) ────────────────────────────────

export function textFromUnknown(value: unknown, depth = 0): string {
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

export function toolPartFromStoredCall(call: unknown, fallbackIndex: number, timestamp?: number): ChatMessagePart {
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
    argsText: Object.keys(args).length ? JSON.stringify(args) : '',
    ...(timestamp !== undefined ? { timestamp } : {})
  }
}

export function applyStoredToolResult(messages: ChatMessage[], toolMessage: SessionMessage): boolean {
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
      completedAt: toolMessage.timestamp,
      result: parseStoredToolResult(content),
      isError: false
    } as ChatMessagePart
    messages[i] = { ...message, parts }

    return true
  }

  return false
}

export function applyStoredToolResultToParts(
  parts: ChatMessagePart[],
  toolMessage: SessionMessage
): ChatMessagePart[] | null {
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
    completedAt: toolMessage.timestamp,
    result: parseStoredToolResult(content),
    isError: false
  } as ChatMessagePart

  return next
}

export function storedToolMessagePart(toolMessage: SessionMessage, fallbackIndex: number): ChatMessagePart {
  const name = toolMessage.tool_name || toolMessage.name || 'tool'
  const context = textFromUnknown(toolMessage.context || toolMessage.text || toolMessage.content || '')
  // Prefer the full arguments when the gateway projection carries them:
  // `context` is an 80-char display preview, and the expanded tool row
  // rebuilds the real command from args. Keep `context` alongside as the
  // title-side placeholder.
  const storedArgs = parseMaybeJsonObject(toolMessage.args)
  const args = { ...storedArgs, ...(context ? { context } : {}) }

  return {
    type: 'tool-call',
    toolCallId: toolMessage.tool_call_id || `stored-tool-message-${fallbackIndex}`,
    toolName: name,
    args: args as never,
    argsText: Object.keys(args).length ? JSON.stringify(args) : '',
    timestamp: toolMessage.timestamp,
    completedAt: toolMessage.timestamp,
    result: context ? { context } : {},
    isError: false
  }
}

export function withUniqueToolCallIds(messages: ChatMessage[]): ChatMessage[] {
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
