import { mediaDisplayLabel, mediaMarkdownHref } from '@/lib/media'

import type { ChatMessage, ChatMessagePart } from './types'

export function textPart(text: string, timestamp?: number): ChatMessagePart {
  return { type: 'text', text, ...(timestamp !== undefined ? { timestamp } : {}) }
}

export function reasoningPart(text: string, timestamp?: number): ChatMessagePart {
  return { type: 'reasoning', text, ...(timestamp !== undefined ? { timestamp } : {}) }
}

/**
 * Known deliverable file extensions — mirrors the Python-side
 * `MEDIA_DELIVERY_EXTS` in `gateway/platforms/base.py` so the two surfaces
 * agree on which `MEDIA:` paths are valid. Used to anchor the end of an
 * unquoted path that may contain interior spaces (#96657).
 */
const MEDIA_DELIVERY_EXTS = [
  'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff', 'svg',
  'mp4', 'mov', 'avi', 'mkv', 'webm', '3gp',
  'mp3', 'm2a', 'wav', 'ogg', 'opus', 'm4a', 'flac',
  'pdf', 'docx', 'doc', 'odt', 'rtf', 'txt', 'md', 'epub',
  'xlsx', 'xls', 'ods', 'csv', 'tsv', 'json', 'xml', 'yaml', 'yml',
  'kmz', 'kml', 'geojson', 'gpx',
  'pptx', 'ppt', 'odp', 'key',
  'zip', 'tar', 'gz', 'tgz', 'bz2', 'xz', '7z', 'rar', 'apk', 'ipa',
  'html', 'htm',
] as const

// Sort longest-first so the alternation never matches a shorter ext as a
// prefix of a longer one (e.g. `tar` before a hypothetical `tar.gz`).
const _MEDIA_EXT_ALTERNATION = [...MEDIA_DELIVERY_EXTS]
  .sort((a, b) => b.length - a.length)
  .join('|')

/**
 * Unquoted path branch: starts with a path anchor (`~/`, `/`, `X:\` or `X:/`),
 * allows interior whitespace, and anchors the end on a known deliverable
 * extension. Matches the Python-side `MEDIA_TAG_CLEANUP_RE` behavior where
 * `(?:[^\S\n]+\S+?)*?\.(?:EXT)` permits spaces inside filenames (#96657).
 */
const _MEDIA_PATH_ANCHORED =
  `(?:~/|/|[A-Za-z]:[/\\\\])\\S+?(?:[^\\S\\n]+\\S+?)*?\\.(?:${_MEDIA_EXT_ALTERNATION})(?=[\\s\`"'_,;:)\\]}]|MEDIA:|$)`

const MEDIA_LINE_RE = new RegExp(
  `(^|\\n)[\\t ]*[\`"']?MEDIA:\\s*(?<line>\`[^\`\\n]+\`|"[^"\\n]+"|'[^'\\n]+'|${_MEDIA_PATH_ANCHORED}|\\S+)[\`"']?[\\t ]*(\\n|$)`,
  'g'
)

const MEDIA_TAG_RE = new RegExp(
  `[\`"']?MEDIA:\\s*(?<inline>\`[^\`\\n]+\`|"[^"\\n]+"|'[^'\\n]+'|${_MEDIA_PATH_ANCHORED}|\\S+)[\`"']?`,
  'g'
)

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
