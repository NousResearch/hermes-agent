import type { ThreadMessageLike } from '@assistant-ui/react'
import { type BillingBlock, skillInvocationText } from '@hermes/shared'

import { extractImageRefs } from '@/lib/embedded-images'
import { dedupeGeneratedImageEchoesInParts } from '@/lib/generated-images'
import { mediaDisplayLabel, mediaMarkdownHref } from '@/lib/media'
import { normalize } from '@/lib/text'
import { parseTodos } from '@/lib/todos'
import type { MessageReaction, SessionMessage, UsageStats } from '@/types/hermes'

export interface TimelinePartMetadata {
  /** Unix seconds when this visible activity segment began. Fractional values
   * preserve the millisecond precision available on live gateway events. */
  timestamp?: number
  /** Unix seconds when this segment stopped or handed off to the next one. */
  completedAt?: number
}

export type ChatMessagePart = Exclude<ThreadMessageLike['content'], string>[number] & TimelinePartMetadata

export type ChatMessage = {
  id: string
  role: SessionMessage['role']
  parts: ChatMessagePart[]
  timestamp?: number
  completedAt?: number
  pending?: boolean
  error?: string
  branchGroupId?: string
  hidden?: boolean
  /** Sealed mid-turn commentary (`message.interim`) — rendered without the
   *  action footer so only the turn's final reply carries copy/refresh, and
   *  the live view matches rehydration (which merges the turn into one bubble). */
  interim?: boolean
  /** Whole-turn wall-clock seconds (message.start → message.complete),
   *  stamped by the desktop when it watched the turn run. Absent for
   *  messages hydrated from history — the backend doesn't persist it. */
  durationS?: number
  /** Composer attachment ref strings (`@file:...`, `@image:...`) sent with this user message. */
  attachmentRefs?: string[]
  /** Durable backend `messages.id`. Absent until the row is persisted. */
  rowId?: number
  /** Emoji reactions on this message — one per author (see MessageReaction). */
  reactions?: MessageReaction[]
}

export type GatewayEventPayload = {
  /** Unix seconds supplied by tests/newer gateways; the desktop falls back to
   * its local receipt clock when older gateways omit it. */
  timestamp?: number
  text?: string
  rendered?: string
  status?: string
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
  approval_mode?: string
  yolo?: boolean
  running?: boolean
  turn_started_at?: number | null
  cwd?: string
  branch?: string
  terminal_backend?: string
  credential_warning?: string
  install_warning?: string
  personality?: string
  usage?: Partial<UsageStats>
  // agent.terminal.output — live chunk for a read-only agent terminal tab
  process_id?: string
  chunk?: string
  // clarify.request
  request_id?: string
  question?: string
  choices?: string[] | null
  multi_select?: boolean
  // clarify.request batch form: questions replaces question/choices, and
  // answers (qid → locked answer) rides along on reconnect replay only.
  questions?: unknown
  answers?: Record<string, unknown>
  // mcp.setup.request (setup_mcp tool — inline MCP consent card)
  server?: string
  action?: string
  reason?: string
  // approval.request (dangerous command / execute_code) — session-keyed
  command?: string
  description?: string
  // False when a tirith content-security warning forbids a permanent allow.
  allow_permanent?: boolean
  smart_denied?: boolean
  // secret.request (skill credential capture)
  env_var?: string
  prompt?: string
  // terminal.read.request / preview.read.request (GUI agent reading the
  // in-app terminal pane or the browser/preview pane)
  start?: number
  count?: number
  // status.update (kind=process → background process completion/watch-match)
  kind?: string
  // pane.reveal (agent focusing a desktop pane via the focus_pane tool)
  pane?: string
  // layout.apply (agent applying a layout preset via the apply_layout tool)
  preset?: string
  // message.reaction (agent reacting via the react_to_message tool) — the
  // durable messages.id, that row's full reaction list after the write, and
  // the row's role so a live (not-yet-round-tripped) message can be matched.
  row_id?: number
  reactions?: MessageReaction[]
  role?: string
  // session.title (live auto-title push) — stored session id + generated title
  session_id?: string
  title?: string
  // session.info — the stored (durable) session id for this runtime session.
  // Lets the desktop app map runtime→stored for background sessions it hasn't
  // opened, so the sidebar working indicator updates without opening the chat.
  stored_session_id?: string
  // moa.reference / moa.aggregating (Mixture of Agents per-model relay)
  label?: string
  index?: number
  aggregator?: string
  // moa.progress / moa.phase (Mixture of Agents fan-out progress relay)
  refs_done?: number
  refs_total?: number
  phase?: string
  // message.complete — signals the final text was already previewed via
  // interim_assistant_callback, so the UI can settle instead of duplicating.
  response_previewed?: boolean
  // message.complete with status "error" — `text` is streamed partial output
  // (keep it visible), not the error string.
  partial?: boolean
  // message.complete with status "error" — the failed turn was retained
  // backend-side and will replay through session.resume's inflight payload.
  recoverable?: boolean
  // Structured billing wall forwarded on message.complete when a turn fails
  // with FailoverReason.billing (shape mirrors @hermes/shared BillingBlock).
  billing?: BillingBlock
  failure_reason?: string
}

export function textPart(text: string, timestamp?: number): ChatMessagePart {
  return { type: 'text', text, ...(timestamp !== undefined ? { timestamp } : {}) }
}

export function reasoningPart(text: string, timestamp?: number): ChatMessagePart {
  return { type: 'reasoning', text, ...(timestamp !== undefined ? { timestamp } : {}) }
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

export function assistantTextPart(text: string, timestamp?: number): ChatMessagePart {
  return textPart(renderMediaTags(text), timestamp)
}

export function chatMessageText(message: ChatMessage): string {
  return message.parts
    .filter((part): part is Extract<ChatMessagePart, { type: 'text' }> => part.type === 'text')
    .map(part => part.text)
    .join('')
}

export interface UnspokenTurnSpeech {
  /** First unspoken assistant bubble — stable for the turn, the live speech session binds to it. */
  id: string
  /** Whether the newest assistant bubble is still streaming. */
  pending: boolean
  /** All unspoken assistant text in message order, bubbles joined on a blank line. */
  text: string
}

/**
 * Collect every unspoken assistant bubble after `lastSpokenId`, in order.
 *
 * A turn with tool calls produces several assistant bubbles — narration
 * ("Let me check…") sealed as interims, then the final answer as a fresh
 * bubble. Voice conversation speaks a turn through ONE live session bound to
 * one response id, so it needs all of that text as a single growing string;
 * selecting only one bubble silently drops everything after it. The blank-line
 * join is a sentence boundary for the server's cutter, so a sealed bubble's
 * tail is flushed as soon as the next bubble starts.
 */
export function collectUnspokenTurnSpeech(
  messages: ChatMessage[],
  lastSpokenId: string | null
): UnspokenTurnSpeech | null {
  const spokenIndex = lastSpokenId ? messages.findLastIndex(m => m.id === lastSpokenId) : -1

  let id: string | null = null
  let pending = false
  const parts: string[] = []

  for (const message of messages.slice(spokenIndex + 1)) {
    if (message.role !== 'assistant' || message.hidden) {
      continue
    }

    pending = Boolean(message.pending)
    const text = chatMessageText(message).trim()

    if (!text) {
      continue
    }

    id ??= message.id
    parts.push(text)
  }

  if (!id) {
    return null
  }

  return { id, pending, text: parts.join('\n\n') }
}

const normalizeWs = (value: string) => value.replace(/\s+/g, ' ').trim()

/**
 * Drop earlier text parts that a later text part repeats verbatim (after
 * whitespace normalization). Providers that continue a turn after a tool
 * call sometimes re-send the previous assistant text as the next message's
 * prefix (tool_calls row, then a stop row with identical prose) — the turn
 * merge then holds the same paragraph twice and everything in it renders
 * twice, most visibly ::preview frames. The LAST occurrence is the
 * authoritative one; keep it.
 */
export function dedupeRepeatedTextInParts(parts: ChatMessagePart[]): ChatMessagePart[] {
  const lastByText = new Map<string, number>()

  parts.forEach((part, index) => {
    if (part.type === 'text') {
      const key = normalizeWs(part.text)

      if (key) {
        lastByText.set(key, index)
      }
    }
  })

  const dropped = parts.filter((part, index) => {
    if (part.type !== 'text') {
      return true
    }

    const key = normalizeWs(part.text)

    return !key || lastByText.get(key) === index
  })

  return dropped.length === parts.length ? parts : dropped
}

/**
 * Merge the final assistant text into a message's parts.
 *
 * - Removes all existing `text` parts (they were streamed deltas, now superseded
 *   by the authoritative final response).
 * - Keeps `reasoning` parts, but drops one that the final text fully covers
 *   (reasoning ⊆ final) — the final restates it. A short final ("Done.") must
 *   NOT swallow a longer reasoning block that merely starts with it (#61447).
 * - Keeps all other part types (tool-call, image, etc.).
 * - Appends the final text as a new text part.
 */
export function mergeFinalAssistantText(
  parts: ChatMessagePart[],
  finalText: string,
  fallbackTimestamp?: number
): ChatMessagePart[] {
  const dedupeReference = normalizeWs(finalText)

  const streamedText = normalizeWs(
    parts
      .filter((part): part is Extract<ChatMessagePart, { type: 'text' }> => part.type === 'text')
      .map(part => part.text)
      .join('')
  )

  // An authoritative final that is exactly the concatenation of streamed text
  // confirms the content without erasing text↔reasoning activity boundaries.
  if (streamedText && streamedText === dedupeReference) {
    return parts
  }

  const previousText = parts.findLast(part => part.type === 'text')

  const kept = parts.filter(part => {
    if (part.type === 'text') {
      // Sealed text parts were already finalized into their own bubbles —
      // this filter only runs on the LAST streaming bubble, so there are no
      // sealed parts here. All text parts are streamed deltas that get
      // replaced by the authoritative final text.
      return false
    }

    if (part.type !== 'reasoning' || !dedupeReference) {
      return true
    }

    // Reasoning is a restatement only when the final FULLY covers it.
    // The reverse direction is not considered — a short final must not
    // swallow a longer reasoning block (#61447).
    const r = normalizeWs(part.text)

    return !(r && dedupeReference.startsWith(r))
  })

  if (!finalText) {
    return kept
  }

  const finalPart = assistantTextPart(finalText, previousText?.timestamp ?? fallbackTimestamp)

  if (previousText?.completedAt !== undefined) {
    finalPart.completedAt = previousText.completedAt
  }

  return [...kept, finalPart]
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

  // A `/skill` turn is stored expanded (the whole skill body). Current
  // gateways project it to the invocation before it ever reaches us; this is
  // the fallback for an older backend that still ships the raw payload.
  const invocation = skillInvocationText(textContent)

  if (invocation) {
    return invocation
  }

  const marker = textContent.match(ATTACHED_CONTEXT_MARKER_RE)

  if (!marker || marker.index === undefined) {
    return textContent.replace(CONTEXT_WARNINGS_MARKER_RE, '').trim()
  }

  const visibleText = textContent.slice(0, marker.index).replace(CONTEXT_WARNINGS_MARKER_RE, '').trim()
  const attachedContext = textContent.slice(marker.index + marker[0].length)
  const refs = [...new Set(Array.from(attachedContext.matchAll(CONTEXT_REF_RE)).map(match => match[0]))]

  // The prose keeps the `@file:` token the user typed, so it already chips in
  // place. Only hoist a ref the prose is missing — a turn persisted by an older
  // backend that stripped the tokens. Re-listing an inline ref would chip twice.
  const missing = refs.filter(ref => !visibleText.includes(ref))

  return [missing.join('\n'), visibleText].filter(Boolean).join('\n\n') || visibleText
}

function transcriptContent(displayKind: SessionMessage['display_kind'], content: string): string | null {
  return displayKind === 'hidden' ? null : content
}

// A remote backend older than this app serves display_metadata as raw JSON text,
// and `in` throws on a primitive — which used to fail the whole session resume.
function parseDisplayMetadata(metadata: SessionMessage['display_metadata']): null | Record<string, unknown> {
  let parsed: unknown = metadata

  if (typeof parsed === 'string') {
    try {
      parsed = JSON.parse(parsed)
    } catch {
      return null
    }
  }

  return parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null
}

function timelineTaskCount(metadata: SessionMessage['display_metadata']): number | undefined {
  const count = parseDisplayMetadata(metadata)?.task_count

  return typeof count === 'number' ? count : undefined
}

export function messageReactions(metadata: SessionMessage['display_metadata']): MessageReaction[] {
  const reactions = parseDisplayMetadata(metadata)?.reactions

  if (!Array.isArray(reactions)) {
    return []
  }

  return reactions.filter(
    (r): r is MessageReaction => Boolean(r) && typeof r === 'object' && typeof (r as MessageReaction).emoji === 'string'
  )
}

function timelineDisplayContent(message: SessionMessage, content: string): string {
  if (message.display_kind === 'model_switch') {
    return 'model changed'
  }

  if (message.display_kind === 'auto_continue') {
    return 'resumed interrupted turn'
  }

  if (message.display_kind === 'personality_switch') {
    return 'personality changed'
  }

  if (message.display_kind === 'async_delegation_complete') {
    const count = timelineTaskCount(message.display_metadata)

    return count === undefined
      ? 'background agent work finished'
      : `${count} background agent${count === 1 ? '' : 's'} finished`
  }

  return content
}

const STREAM_PART: Record<'reasoning' | 'text', (text: string, timestamp?: number) => ChatMessagePart> = {
  reasoning: reasoningPart,
  text: textPart
}

function completeOpenStreamParts(parts: ChatMessagePart[], completedAt: number): ChatMessagePart[] {
  return parts.map(part =>
    (part.type === 'text' || part.type === 'reasoning') && part.completedAt === undefined
      ? ({ ...part, completedAt } as ChatMessagePart)
      : part
  )
}

/** Seal every still-open visible activity when the assistant turn stops. */
export function completeOpenTimelineParts(parts: ChatMessagePart[], completedAt: number): ChatMessagePart[] {
  return parts.map(part =>
    part.timestamp !== undefined && part.completedAt === undefined
      ? ({ ...part, completedAt } as ChatMessagePart)
      : part
  )
}

// Coalesce only adjacent deltas of the same channel. Switching between text
// and reasoning is a real timeline boundary and must remain visible even when
// both channels arrive inside one batched renderer flush.
function appendStreamPart(
  parts: ChatMessagePart[],
  type: 'reasoning' | 'text',
  delta: string,
  timestamp?: number
): { index: number; parts: ChatMessagePart[] } {
  const next = [...parts]

  const tailIndex = next.length - 1
  const tail = next[tailIndex]

  if (tail?.type === type && tail.completedAt === undefined) {
    next[tailIndex] = { ...tail, text: `${tail.text}${delta}` } as ChatMessagePart

    return { index: tailIndex, parts: next }
  }

  if (
    timestamp !== undefined &&
    (tail?.type === 'text' || tail?.type === 'reasoning') &&
    tail.completedAt === undefined
  ) {
    next[tailIndex] = { ...tail, completedAt: timestamp } as ChatMessagePart
  }

  next.push(STREAM_PART[type](delta, timestamp))

  return { index: next.length - 1, parts: next }
}

export function appendTextPart(parts: ChatMessagePart[], delta: string, timestamp?: number): ChatMessagePart[] {
  return appendStreamPart(parts, 'text', delta, timestamp).parts
}

export function appendReasoningPart(parts: ChatMessagePart[], delta: string, timestamp?: number): ChatMessagePart[] {
  return appendStreamPart(parts, 'reasoning', delta, timestamp).parts
}

export function appendAssistantTextPart(
  parts: ChatMessagePart[],
  delta: string,
  timestamp?: number
): ChatMessagePart[] {
  const { index, parts: next } = appendStreamPart(parts, 'text', delta, timestamp)
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

function toolPartFromStoredCall(call: unknown, fallbackIndex: number, timestamp?: number): ChatMessagePart {
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
      completedAt: toolMessage.timestamp,
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
    completedAt: toolMessage.timestamp,
    result: parseStoredToolResult(content),
    isError: false
  } as ChatMessagePart

  return next
}

function storedToolMessagePart(toolMessage: SessionMessage, fallbackIndex: number): ChatMessagePart {
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

  const earliestTimestamp = (...values: (number | undefined)[]) => {
    const timestamps = values.filter((value): value is number => value !== undefined)

    return timestamps.length ? Math.min(...timestamps) : undefined
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
    active.timestamp = earliestTimestamp(active.timestamp, timestamp, ...parts.map(part => part.timestamp))

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

    const rawDisplayContent = transcriptContent(
      message.display_kind,
      timelineDisplayContent(message, displayContentForMessage(message.role, content))
    )

    const displayRole =
      message.display_kind === 'model_switch' ||
      message.display_kind === 'async_delegation_complete' ||
      message.display_kind === 'auto_continue' ||
      message.display_kind === 'personality_switch'
        ? 'system'
        : message.role

    // Persisted user turns carry `@image:<path>` directive lines inline in
    // the text (see tui_gateway/server.py's persist-time rewrite). The
    // read-only bubble clamps its body to ~2 lines, and a large inline image
    // thumbnail pushes any caption text below the clamp's visible area — so
    // pull image refs out into `attachmentRefs` (same shape the local
    // optimistic composer already uses) and render them via the dedicated
    // attachments row below the bubble instead.
    const imageRefExtraction = displayRole === 'user' && rawDisplayContent ? extractImageRefs(rawDisplayContent) : null
    const displayContent = imageRefExtraction ? imageRefExtraction.cleanedText : rawDisplayContent
    const extractedAttachmentRefs = imageRefExtraction?.refs.length ? imageRefExtraction.refs : undefined

    const parts: ChatMessagePart[] = []

    const reasoning =
      message.reasoning ||
      message.reasoning_content ||
      (typeof message.reasoning_details === 'string' ? message.reasoning_details : '')

    if (reasoning && message.role === 'assistant') {
      parts.push(reasoningPart(reasoning, message.timestamp))
    }

    if (displayContent) {
      parts.push(
        displayRole === 'assistant'
          ? assistantTextPart(displayContent, message.timestamp)
          : textPart(displayContent, message.timestamp)
      )
    }

    if (message.role === 'assistant' && Array.isArray(message.tool_calls)) {
      parts.push(
        ...message.tool_calls.map((call, callIndex) => toolPartFromStoredCall(call, callIndex, message.timestamp))
      )
    }

    if (!parts.length && !extractedAttachmentRefs?.length) {
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
        activeAssistant.timestamp = earliestTimestamp(
          activeAssistant.timestamp,
          message.timestamp,
          ...parts.map(part => part.timestamp)
        )

        return
      }
    } else {
      flushPendingTools(index)
    }

    const reactions = messageReactions(message.display_metadata)
    // Gateway resume names the durable row id `row_id`; the REST transcript
    // prefetch ships the same messages.id as a numeric `id`. Either one lets
    // reactions address this exact row later.
    const rowId = message.row_id ?? (typeof message.id === 'number' ? message.id : undefined)

    result.push({
      id: `${message.timestamp || Date.now()}-${index}-${displayRole}`,
      role: displayRole,
      parts,
      timestamp: earliestTimestamp(message.timestamp, ...parts.map(part => part.timestamp)),
      ...(rowId !== undefined ? { rowId } : {}),
      ...(reactions.length ? { reactions } : {}),
      ...(extractedAttachmentRefs ? { attachmentRefs: extractedAttachmentRefs } : {})
    })

    activeAssistantIndex = message.role === 'assistant' ? result.length - 1 : null
  })
  flushPendingTools(messages.length)

  const withoutGeneratedImageEchoes = result.map(message =>
    message.role === 'assistant'
      ? { ...message, parts: dedupeRepeatedTextInParts(dedupeGeneratedImageEchoesInParts(message.parts)) }
      : message
  )

  return withUniqueToolCallIds(
    withoutGeneratedImageEchoes.filter(
      m => chatMessageText(m).trim() || m.parts.some(part => part.type !== 'text') || m.attachmentRefs?.length
    )
  )
}

const validTimelineBoundary = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0

const earliestBoundary = (...values: (number | undefined)[]) => {
  const valid = values.filter(validTimelineBoundary)

  return valid.length ? Math.min(...valid) : undefined
}

const latestBoundary = (...values: (number | undefined)[]) => {
  const valid = values.filter(validTimelineBoundary)

  return valid.length ? Math.max(...valid) : undefined
}

const normalizedTimelineText = (message: ChatMessage) => chatMessageText(message).replace(/\s+/g, ' ').trim()

const assistantTimelineMatch = (stored: ChatMessage, local: ChatMessage) => {
  if (stored.id === local.id) {
    return true
  }

  const localToolIds = new Set(
    local.parts
      .filter(part => part.type === 'tool-call')
      .map(part => (part.type === 'tool-call' ? part.toolCallId : ''))
  )

  const toolMatch = stored.parts.some(part => part.type === 'tool-call' && localToolIds.has(part.toolCallId))

  if (toolMatch) {
    return true
  }

  const storedText = normalizedTimelineText(stored)

  return Boolean(storedText) && storedText === normalizedTimelineText(local)
}

const timelinePartMatch = (stored: ChatMessagePart, local: ChatMessagePart) => {
  if (stored.type !== local.type) {
    return false
  }

  if (stored.type === 'tool-call' && local.type === 'tool-call') {
    return stored.toolCallId === local.toolCallId
  }

  if ((stored.type === 'text' || stored.type === 'reasoning') && local.type === stored.type) {
    return stored.text.replace(/\s+/g, ' ').trim() === local.text.replace(/\s+/g, ' ').trim()
  }

  return false
}

/** Keep richer live timing when durable hydration has only one timestamp per row. */
export function reconcileLocalAssistantTimeline(
  nextMessages: ChatMessage[],
  currentMessages: ChatMessage[]
): ChatMessage[] {
  const localAssistants = currentMessages.filter(message => message.role === 'assistant' && !message.hidden)
  const matches = new Map<number, ChatMessage>()
  let localCursor = localAssistants.length - 1

  for (let nextIndex = nextMessages.length - 1; nextIndex >= 0; nextIndex -= 1) {
    const message = nextMessages[nextIndex]

    if (message.role !== 'assistant' || message.hidden) {
      continue
    }

    for (let localIndex = localCursor; localIndex >= 0; localIndex -= 1) {
      const local = localAssistants[localIndex]

      if (assistantTimelineMatch(message, local)) {
        matches.set(nextIndex, local)
        localCursor = localIndex - 1

        break
      }
    }
  }

  return nextMessages.map((message, messageIndex) => {
    const local = matches.get(messageIndex)

    if (!local) {
      return message
    }

    const unusedLocalParts = new Set(local.parts.map((_, index) => index))

    const parts = message.parts.map(part => {
      const localIndex = local.parts.findIndex(
        (candidate, index) => unusedLocalParts.has(index) && timelinePartMatch(part, candidate)
      )

      if (localIndex === -1) {
        return part
      }

      unusedLocalParts.delete(localIndex)
      const localPart = local.parts[localIndex]

      return {
        ...part,
        completedAt: latestBoundary(part.completedAt, localPart.completedAt),
        timestamp: earliestBoundary(part.timestamp, localPart.timestamp)
      } as ChatMessagePart
    })

    return {
      ...message,
      completedAt: latestBoundary(message.completedAt, local.completedAt, ...parts.map(part => part.completedAt)),
      parts,
      timestamp: earliestBoundary(message.timestamp, local.timestamp, ...parts.map(part => part.timestamp))
    }
  })
}

export function preserveLocalAssistantErrors(
  nextMessages: ChatMessage[],
  currentMessages: ChatMessage[]
): ChatMessage[] {
  nextMessages = reconcileLocalAssistantTimeline(nextMessages, currentMessages)
  const localById = new Map(currentMessages.map(message => [message.id, message]))

  const mergedNextMessages = nextMessages.map(message => {
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

  for (let index = 0; index < currentMessages.length; index += 1) {
    const message = currentMessages[index]

    if (message.role !== 'assistant' || !message.error || message.hidden || existingIds.has(message.id)) {
      continue
    }

    preserveIds.add(message.id)

    for (let probe = index - 1; probe >= 0; probe -= 1) {
      const candidate = currentMessages[probe]

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

  const preserved = currentMessages
    .filter(message => preserveIds.has(message.id))
    .map(message => ({ ...message, pending: false }))

  return [...mergedNextMessages, ...preserved]
}

export function branchGroupForUser(userMessage: ChatMessage): string {
  return `branch:${userMessage.id}`
}
