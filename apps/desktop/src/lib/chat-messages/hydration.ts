import { skillInvocationText } from '@hermes/shared'

import { extractImageRefs } from '@/lib/embedded-images'
import { dedupeGeneratedImageEchoesInParts } from '@/lib/generated-images'
import type { MessageReaction, SessionMessage } from '@/types/hermes'

import { assistantTextPart, chatMessageText, dedupeRepeatedTextInParts, reasoningPart, textPart } from './parts'
import {
  applyStoredToolResult,
  applyStoredToolResultToParts,
  storedToolMessagePart,
  textFromUnknown,
  toolPartFromStoredCall,
  withUniqueToolCallIds
} from './tool-parts'
import type { ChatMessage, ChatMessagePart } from './types'

const ATTACHED_CONTEXT_MARKER_RE = /(?:^|\n)--- Attached Context ---\s*\n/
const CONTEXT_WARNINGS_MARKER_RE = /(?:^|\n)--- Context Warnings ---[\s\S]*$/
const CONTEXT_REF_RE = /@(file|folder|url|image|tool|terminal):(?:"[^"\n]+"|'[^'\n]+'|`[^`\n]+`|\S+)/g

// Gateway routing note for Discord turns (gateway/run_inbound.py::discord_triggering_note).
// Current gateways persist the authored text; this heals rows written before that fix. Only
// the note is model-facing — the `[Replying to: …]` pointer next to it is kept.
const DISCORD_TRIGGERING_NOTE_RE =
  /(^|\n)\[Triggering message id: `[^`\n]*` — use as `message_id` for reply\/react\/pin via the discord tools\.\]\n*/

/**
 * Reply text from a Responses-API `codex_message_items` sidecar (#68321), for rows
 * whose `content` persisted empty. `commentary` / `analysis` items are mid-turn
 * narration the backend routes to the reasoning channel
 * (codex_responses_adapter `_OutputScan._message`); the remaining phases are the reply.
 */
function codexMessageItemText(message: SessionMessage): string {
  let items = message.codex_message_items

  // REST carries SQLite JSON text; RPC history carries the decoded list.
  if (typeof items === 'string') {
    try {
      items = JSON.parse(items)
    } catch {
      return ''
    }
  }

  if (!Array.isArray(items)) {
    return ''
  }

  const texts: string[] = []

  for (const item of items) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      continue
    }

    const record = item as Record<string, unknown>

    if (record.type !== 'message' || record.role !== 'assistant') {
      continue
    }

    if (record.phase === 'commentary' || record.phase === 'analysis') {
      continue
    }

    const content = record.content

    if (!Array.isArray(content)) {
      continue
    }

    for (const part of content) {
      if (!part || typeof part !== 'object' || Array.isArray(part)) {
        continue
      }

      const partRecord = part as Record<string, unknown>
      const partType = partRecord.type

      if (partType !== 'output_text' && partType !== 'text') {
        continue
      }

      const text = partRecord.text

      if (typeof text === 'string' && text.length > 0) {
        texts.push(text)
      }
    }
  }

  return texts.join('')
}

function displayContentForMessage(role: SessionMessage['role'], content: unknown): string {
  const rawText = textFromUnknown(content)

  if (role !== 'user') {
    return rawText
  }

  const textContent = rawText.replace(DISCORD_TRIGGERING_NOTE_RE, '$1')

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

function timelineDisplayText(metadata: SessionMessage['display_metadata']): string | undefined {
  const text = parseDisplayMetadata(metadata)?.display_text

  return typeof text === 'string' && text.trim() ? text : undefined
}

function messageReactions(metadata: SessionMessage['display_metadata']): MessageReaction[] {
  const reactions = parseDisplayMetadata(metadata)?.reactions

  if (!Array.isArray(reactions)) {
    return []
  }

  return reactions.filter(
    (r): r is MessageReaction => Boolean(r) && typeof r === 'object' && typeof (r as MessageReaction).emoji === 'string'
  )
}

// Only parse producer-owned boundaries, never render the model's task preamble.
// Older backends can persist an unwrapped result rather than an envelope.
function asyncResultBody(content: string): string | undefined {
  let bodies = [content]

  if (content.startsWith('[IMPORTANT: ')) {
    // Background-process completion: one `[IMPORTANT: …]` block per process, a batch header first.
    bodies = content
      .split(/\n\n(?=\[IMPORTANT: )/)
      .map(block => block.replace(/^\[IMPORTANT:\s*/, '').replace(/\]$/, ''))
      .filter(block => !/^\d+ background processes completed\./.test(block))
  } else if (content.startsWith('[ASYNC DELEGATION')) {
    if (content.startsWith('[ASYNC DELEGATION BATCH COMPLETE')) {
      // Task goals can span lines; stopping at a newline leaks the next goal and transcript footer.
      bodies = content.split(/^--- [✓✗⚠] TASK \d+\/\d+(?:: [\s\S]*?)? {2}\(status=[^\n]*\) ---\r?\n/gm).slice(1)
    } else {
      const result = content.match(/^--- (?:RESULT|ERROR) ---\r?\n/m)
      bodies = result ? [content.slice(result.index! + result[0].length)] : []
    }
  }

  return (
    bodies
      .map(body => {
        const output = body.startsWith('Cron job ') ? body.match(/^--- JOB OUTPUT ---\r?\n/m) : null
        const result = output ? body.slice(output.index! + output[0].length) : body

        return result.replace(/\nFull live transcript \(complete tool\/assistant trace\): [^\n]*\n*$/, '').trim()
      })
      .filter(Boolean)
      .join('\n\n') || undefined
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

    return (
      timelineDisplayText(message.display_metadata) ??
      (count === undefined
        ? 'background agent work finished'
        : `${count} background agent${count === 1 ? '' : 's'} finished`)
    )
  }

  if (message.display_kind === 'process_complete') {
    return timelineDisplayText(message.display_metadata) ?? 'background process finished'
  }

  return content
}

/**
 * An assistant row with NO visible content — empty/blank `content` (and no
 * codex sidecar, no tool_calls) — but non-empty `reasoning` is a retry-churn
 * artifact, not a reply. It must never enter the merge loop: `toChatMessages`
 * would otherwise graft its reasoning onto a retained text bubble via the
 * tool-attach path, and a post-merge filter cannot see it (the bubble still
 * has text). Reasoning is a private scratchpad, not a substitute for an
 * answer — drop these rows before merging.
 */
function isDropOnlyReasoningAssistant(message: SessionMessage): boolean {
  if (message.role !== 'assistant') {
    return false
  }

  const visibleContent =
    message.display_content !== undefined
      ? message.display_content
      : message.content || message.text || message.context || message.name

  if (visibleContent && String(visibleContent).trim()) {
    return false
  }

  if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
    return false
  }

  if (codexMessageItemText(message)) {
    return false
  }

  const reasoning =
    message.reasoning ||
    message.reasoning_content ||
    (typeof message.reasoning_details === 'string' ? message.reasoning_details : '')

  return Boolean(reasoning)
}

export function toChatMessages(messages: SessionMessage[]): ChatMessage[] {
  const result: ChatMessage[] = []
  let pendingToolParts: ChatMessagePart[] = []
  let pendingToolTimestamp: number | undefined
  // Backend rows the pending batch stands for. The fold merges a turn's tool
  // rows into one message, and the store's older-page offset is counted in
  // backend rows, so the folded message has to report how many it covers
  // (see ChatMessage.serverRowSpan).
  let pendingToolRows = 0
  let activeAssistantIndex: null | number = null

  // Drop retry-churn artifacts BEFORE merging: a reasoning-only row that
  // enters the loop can graft its reasoning onto a retained text bubble via
  // the tool-attach path (`pendingToolParts` / `activeAssistant`), and the
  // end filter cannot see it because the bubble still has text.
  const visibleRows = messages.filter(message => !isDropOnlyReasoningAssistant(message))

  const clearPendingTools = () => {
    pendingToolParts = []
    pendingToolTimestamp = undefined
    pendingToolRows = 0
  }

  /** Attribute `rows` backend rows to a folded message (absent field means one). */
  const absorbRows = (message: ChatMessage | undefined, rows: number) => {
    if (message && rows > 0) {
      message.serverRowSpan = (message.serverRowSpan ?? 1) + rows
    }
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
    active.durableComplete = false
    active.timestamp = earliestTimestamp(active.timestamp, timestamp, ...parts.map(part => part.timestamp))
    absorbRows(active, pendingToolRows)

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
        durableComplete: false,
        ...(pendingToolRows > 1 ? { serverRowSpan: pendingToolRows } : {}),
        timestamp: pendingToolTimestamp
      })
      activeAssistantIndex = result.length - 1
    }

    clearPendingTools()
  }

  visibleRows.forEach((message, index) => {
    if (message.role === 'tool') {
      const updatedPendingToolParts = applyStoredToolResultToParts(pendingToolParts, message)

      if (updatedPendingToolParts) {
        pendingToolParts = updatedPendingToolParts
        pendingToolRows += 1

        return
      }

      if (applyStoredToolResult(result, message)) {
        return
      }

      pendingToolParts = [...pendingToolParts, storedToolMessagePart(message, index)]
      pendingToolTimestamp ??= message.timestamp
      pendingToolRows += 1

      return
    }

    const content =
      message.display_content !== undefined
        ? message.display_content
        : message.content || message.text || message.context || message.name

    const rawDisplayContent = transcriptContent(
      message.display_kind,
      timelineDisplayContent(message, displayContentForMessage(message.role, content))
    )

    const displayRole =
      message.display_kind === 'model_switch' ||
      message.display_kind === 'async_delegation_complete' ||
      message.display_kind === 'process_complete' ||
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
    const rowId = message.row_id ?? (typeof message.id === 'number' ? message.id : undefined)
    const sourceHasTools = Array.isArray(message.tool_calls) && message.tool_calls.length > 0
    const durableComplete = sourceHasTools ? false : rowId !== undefined ? true : undefined

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

    // Reply text can live only in the sidecar alongside reasoning or tool parts.
    // Those parts are not a substitute for the answer; canonical content still wins.
    if (message.role === 'assistant' && message.display_kind !== 'hidden' && !displayContent) {
      const codexText = codexMessageItemText(message)

      if (codexText) {
        parts.push(assistantTextPart(codexText, message.timestamp))
      }
    }

    if (message.role === 'assistant' && Array.isArray(message.tool_calls)) {
      parts.push(
        ...message.tool_calls.map((call, callIndex) =>
          toolPartFromStoredCall(call, callIndex, message.timestamp, message.tool_call_labels)
        )
      )
    }

    if (rowId !== undefined) {
      for (const part of parts) {
        if (part.type === 'text') {
          part.sourceRowId = rowId
        }
      }
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
      pendingToolRows += 1

      return
    }

    let pendingAbsorbedRows = 0

    if (message.role === 'assistant') {
      if (pendingToolParts.length) {
        if (!appendPartsToActiveAssistant(pendingToolParts, message.timestamp ?? pendingToolTimestamp)) {
          parts.unshift(...pendingToolParts)
          pendingAbsorbedRows = pendingToolRows
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
        activeAssistant.durableComplete = durableComplete
        activeAssistant.timestamp = earliestTimestamp(
          activeAssistant.timestamp,
          message.timestamp,
          ...parts.map(part => part.timestamp)
        )
        absorbRows(activeAssistant, 1)

        return
      }
    } else {
      flushPendingTools(index)
    }

    const reactions = messageReactions(message.display_metadata)
    // Gateway resume names the durable row id `row_id`; the REST transcript
    // prefetch ships the same messages.id as a numeric `id`. Either one lets
    // reactions address this exact row later.
    result.push({
      id: `${message.timestamp || Date.now()}-${index}-${displayRole}`,
      role: displayRole,
      parts,
      ...(message.role === 'assistant' && durableComplete !== undefined ? { durableComplete } : {}),
      ...(message.display_kind === 'async_delegation_complete' || message.display_kind === 'process_complete'
        ? { asyncResult: asyncResultBody(displayContentForMessage(message.role, message.content || content)) }
        : {}),
      ...(message.display_kind === 'process_complete' ? { asyncResultKind: 'process' as const } : {}),
      timestamp: earliestTimestamp(message.timestamp, ...parts.map(part => part.timestamp)),
      ...(rowId !== undefined ? { rowId } : {}),
      ...(pendingAbsorbedRows > 0 ? { serverRowSpan: pendingAbsorbedRows + 1 } : {}),
      ...(reactions.length ? { reactions } : {}),
      ...(extractedAttachmentRefs ? { attachmentRefs: extractedAttachmentRefs } : {})
    })

    activeAssistantIndex = message.role === 'assistant' ? result.length - 1 : null
  })
  flushPendingTools(visibleRows.length)

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
