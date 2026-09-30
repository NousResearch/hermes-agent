import type { MediaAttachment } from '@hermes/shared'

import { generatedImageEchoSources } from '@/lib/generated-images'

import type { ChatMessage, ChatMessagePart } from './types'

export function textPart(text: string, timestamp?: number): ChatMessagePart {
  return { type: 'text', text, ...(timestamp !== undefined ? { timestamp } : {}) }
}

export function reasoningPart(text: string, timestamp?: number): ChatMessagePart {
  return { type: 'reasoning', text, ...(timestamp !== undefined ? { timestamp } : {}) }
}

/** A file a `MEDIA:` tag delivered. The gateway parses the tag once and sends
 *  it as an attachment beside clean text; the transcript shows it as a card. */
export const ATTACHMENT_PART = 'attachment'

export type AttachmentPart = Extract<ChatMessagePart, { type: 'data' }> & {
  data: MediaAttachment
  name: typeof ATTACHMENT_PART
}

export function attachmentPart(attachment: MediaAttachment): ChatMessagePart {
  return { type: 'data', name: ATTACHMENT_PART, data: { path: attachment.path } }
}

export function isAttachmentPart(part: ChatMessagePart): part is AttachmentPart {
  return part.type === 'data' && part.name === ATTACHMENT_PART
}

/** Add each delivered file once, after everything already in `parts`. A file
 *  that echoes a generated image stays in the tool slot only. */
export function withAttachmentParts(
  parts: ChatMessagePart[],
  attachments: readonly MediaAttachment[] | null | undefined
): ChatMessagePart[] {
  if (!attachments?.length) {
    return parts
  }

  const shown = new Set([...parts.filter(isAttachmentPart).map(part => part.data.path), ...generatedImageEchoSources(parts)])
  const fresh: ChatMessagePart[] = []

  for (const attachment of attachments) {
    if (!shown.has(attachment.path)) {
      shown.add(attachment.path)
      fresh.push(attachmentPart(attachment))
    }
  }

  return fresh.length ? [...parts, ...fresh] : parts
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
 *
 * If `lastSpokenId` is missing or stale (session id assigned mid-turn,
 * live-tail rewrite missed), do **not** fall back to index -1 — that replays
 * every earlier assistant turn as one speech string. Bound to the current
 * turn (assistant bubbles after the last user message) instead. Hidden user
 * rows count: a widget intent (`display_kind: hidden`) is a real turn for the
 * agent even though no bubble renders. A slice with no user row (mid-turn
 * interims only) still collects those assistants.
 */
export function collectUnspokenTurnSpeech(
  messages: ChatMessage[],
  lastSpokenId: string | null
): UnspokenTurnSpeech | null {
  let spokenIndex = lastSpokenId ? messages.findLastIndex(m => m.id === lastSpokenId) : -1

  if (spokenIndex < 0) {
    const lastUser = messages.findLastIndex(m => m.role === 'user')

    if (lastUser >= 0) {
      spokenIndex = lastUser
    }
  }

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

export const normalizeWs = (value: string) => value.replace(/\s+/g, ' ').trim()

type TextPart = Extract<ChatMessagePart, { type: 'text' }>
const isTextPart = (part: ChatMessagePart): part is TextPart => part.type === 'text'

/** The same physical row delivered twice is one occurrence; keep its last copy. */
function dedupeRepeatedRowText(parts: ChatMessagePart[]): ChatMessagePart[] {
  const occurrence = (part: TextPart) => `${part.sourceRowId}:${normalizeWs(part.text)}`
  const lastByOccurrence = new Map<string, number>()

  parts.forEach((part, index) => {
    if (part.type === 'text' && part.sourceRowId !== undefined) {
      lastByOccurrence.set(occurrence(part), index)
    }
  })

  const kept = parts.filter(
    (part, index) =>
      part.type !== 'text' || part.sourceRowId === undefined || lastByOccurrence.get(occurrence(part)) === index
  )

  return kept.length === parts.length ? parts : kept
}

/**
 * Collapse duplicate deliveries of the same text without touching authored
 * repeats. Providers that continue a turn after a tool call sometimes re-send
 * the previous assistant text verbatim as the stop row (tool_calls row, then a
 * stop row with identical prose) — the turn merge then holds the same
 * paragraph twice and everything in it renders twice. Only that shape folds across rows: the bubble's final text (no tool
 * call after it) equal to the text directly before it across a tool call.
 * Equal commentary in earlier tool rounds is authored twice and must hydrate
 * in step with the live stream.
 */
export function dedupeRepeatedTextInParts(parts: ChatMessagePart[]): ChatMessagePart[] {
  const rowDeduped = dedupeRepeatedRowText(parts)
  const texts = rowDeduped.flatMap((part, index) => (isTextPart(part) ? [{ index, part }] : []))
  const [previous, last] = texts.slice(-2)

  if (!last || !previous || rowDeduped.slice(last.index + 1).some(part => part.type === 'tool-call')) {
    return rowDeduped
  }

  const key = normalizeWs(last.part.text)

  if (
    !key ||
    key !== normalizeWs(previous.part.text) ||
    !rowDeduped.slice(previous.index + 1, last.index).some(part => part.type === 'tool-call')
  ) {
    return rowDeduped
  }

  return rowDeduped.filter((_, index) => index !== previous.index)
}

/**
 * Merge the final assistant text into a message's parts.
 *
 * - Preserves earlier tool-delimited responses: a missed interim frame must
 *   not make their public text disposable.
 * - Replaces provisional text only in the latest response with its authoritative
 *   final text, retaining confirmed text/reasoning boundaries.
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
  // Empty / whitespace-only completion is not authoritative — keep streamed
  // text, reasoning, and tool parts (#95514).
  if (!finalText.trim()) {
    return parts
  }

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

  // A tool call is an explicit model-response boundary even when no
  // message.interim frame sealed the earlier text into a separate bubble.
  // Only the suffix after the last call belongs to this authoritative final.
  const lastToolIndex = parts.findLastIndex(part => part.type === 'tool-call')

  if (lastToolIndex >= 0) {
    const earlier = parts.slice(0, lastToolIndex + 1)

    const earlierText = earlier
      .filter((part): part is Extract<ChatMessagePart, { type: 'text' }> => part.type === 'text')
      .map(part => part.text)
      .join('')

    // Some terminal frames carry cumulative text. Strip only an exact prefix;
    // fuzzy similarity is not proof that two assistant messages are the same.
    const responseText =
      earlierText && finalText.startsWith(earlierText) ? finalText.slice(earlierText.length) : finalText

    const suffix = parts.slice(lastToolIndex + 1)

    // A cumulative final can stop exactly at the pre-tool update. The suffix
    // draft is still provisional; the ordinary empty-final path keeps drafts.
    if (earlierText && finalText === earlierText) {
      return [...earlier, ...suffix.filter(part => part.type !== 'text')]
    }

    return [...earlier, ...mergeFinalAssistantText(suffix, responseText, fallbackTimestamp)]
  }

  const previousText = parts.findLast(part => part.type === 'text')

  const kept = parts.filter(part => {
    if (part.type === 'text') {
      // The tool-delimited prefix was retained above. This suffix is
      // provisional text from the response being finalized.
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

  const finalPart = textPart(finalText, previousText?.timestamp ?? fallbackTimestamp)

  if (previousText?.completedAt !== undefined) {
    finalPart.completedAt = previousText.completedAt
  }

  // Delivered files stay after the text of their response.
  return [...kept.filter(part => !isAttachmentPart(part)), finalPart, ...kept.filter(isAttachmentPart)]
}

/** Seal every still-open visible activity when the assistant turn stops. */
export function completeOpenTimelineParts(parts: ChatMessagePart[], completedAt: number): ChatMessagePart[] {
  return parts.map(part =>
    part.timestamp !== undefined && part.completedAt === undefined
      ? ({ ...part, completedAt } as ChatMessagePart)
      : part
  )
}

/** Settle a turn that ended without its terminal message: drop empty
 *  pending/stream placeholders and un-pend the rest. Shared by Stop, the
 *  running=false edge, and the store's silent-turn settle. */
export function finalizeInterruptedMessages(
  messages: ChatMessage[],
  streamId?: null | string,
  occurredAt = Date.now() / 1000
): ChatMessage[] {
  return messages
    .filter(
      message =>
        !(
          (message.pending || message.id === streamId) &&
          message.parts.length === 0 &&
          !chatMessageText(message).trim()
        )
    )
    .map(message =>
      message.pending || message.id === streamId
        ? {
            ...message,
            completedAt: occurredAt,
            parts: completeOpenTimelineParts(message.parts, occurredAt),
            pending: false
          }
        : message
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

  const STREAM_PART: Record<'reasoning' | 'text', (text: string, timestamp?: number) => ChatMessagePart> = {
    reasoning: reasoningPart,
    text: textPart
  }

  next.push(STREAM_PART[type](delta, timestamp))

  return { index: next.length - 1, parts: next }
}

export function appendReasoningPart(parts: ChatMessagePart[], delta: string, timestamp?: number): ChatMessagePart[] {
  return appendStreamPart(parts, 'reasoning', delta, timestamp).parts
}

export function appendAssistantTextPart(
  parts: ChatMessagePart[],
  delta: string,
  timestamp?: number
): ChatMessagePart[] {
  // Delivered files trail the text of their response, live as in history:
  // text streamed after a card joins the text before it.
  const cards = parts.length - (parts.findLastIndex(part => !isAttachmentPart(part)) + 1)

  if (!cards) {
    return appendStreamPart(parts, 'text', delta, timestamp).parts
  }

  return [...appendStreamPart(parts.slice(0, -cards), 'text', delta, timestamp).parts, ...parts.slice(-cards)]
}

/** True when a visible user message follows `messageId` — the reader has moved
 *  on, so a question card at `messageId` counts as answered. */
export function answeredAfter(messages: ChatMessage[], messageId: string): boolean {
  const at = messages.findIndex(message => message.id === messageId)

  return at !== -1 && messages.slice(at + 1).some(message => message.role === 'user' && !message.hidden)
}
