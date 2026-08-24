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

  if (content.startsWith('[ASYNC DELEGATION')) {
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

    return count === undefined
      ? 'background agent work finished'
      : `${count} background agent${count === 1 ? '' : 's'} finished`
  }

  return content
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
      ...(message.display_kind === 'async_delegation_complete'
        ? { asyncResult: asyncResultBody(displayContentForMessage(message.role, message.content || content)) }
        : {}),
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
