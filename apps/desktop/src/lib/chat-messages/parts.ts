import { mediaDisplayLabel, mediaMarkdownHref } from '@/lib/media'

import type { SpokenReplyAnchor } from '../spoken-reply'

import type { ChatMessage, ChatMessagePart } from './types'

export function textPart(text: string, timestamp?: number): ChatMessagePart {
  return { type: 'text', text, ...(timestamp !== undefined ? { timestamp } : {}) }
}

export function reasoningPart(text: string, timestamp?: number): ChatMessagePart {
  return { type: 'reasoning', text, ...(timestamp !== undefined ? { timestamp } : {}) }
}

/** Extract display text from a provider reasoning-details envelope. */
export function reasoningTextFromDetails(details: unknown): string {
  let blocks = details

  if (typeof blocks === 'string') {
    const trimmed = blocks.trim()

    // Some persisted rows contain plain reasoning text rather than a JSON envelope.
    if (!trimmed.startsWith('[') && !trimmed.startsWith('{')) {
      return trimmed
    }

    try {
      blocks = JSON.parse(trimmed) as unknown
    } catch {
      // A malformed structured envelope is replay metadata, not display text.
      return ''
    }
  }

  const text: string[] = []

  const push = (value: unknown) => {
    if (typeof value !== 'string') {
      return
    }

    const prose = value.trim()

    if (prose && !text.includes(prose)) {
      text.push(prose)
    }
  }

  // Nested carrier internals: only genuinely readable reasoning kinds. A
  // native `.native_assistant` carrier wraps signed thinking plus the public
  // answer inside `messages[].content[]`; its `text` blocks are the answer, not
  // reasoning, and signatures/projections/data are opaque replay fields.
  const walkNested = (node: unknown): void => {
    if (Array.isArray(node)) {
      for (const item of node) {
        walkNested(item)
      }

      return
    }

    if (!node || typeof node !== 'object') {
      return
    }

    const record = node as Record<string, unknown>
    const kind = typeof record.type === 'string' ? record.type : ''

    if (kind === 'reasoning.summary') {
      push(record.summary)
    } else if (kind === 'reasoning.text') {
      push(record.text)
    } else if (typeof record.thinking === 'string') {
      push(record.thinking)
    }

    for (const [key, value] of Object.entries(record)) {
      if (key === 'signature' || key === 'projection' || key === 'data' || key === 'type') {
        continue
      }

      if (value && typeof value === 'object') {
        walkNested(value)
      }
    }
  }

  for (const block of Array.isArray(blocks) ? blocks : [blocks]) {
    if (typeof block === 'string') {
      push(block)

      continue
    }

    if (!block || typeof block !== 'object') {
      continue
    }

    // Top-level blocks are provider reasoning-detail entries; recognized
    // prose fields (summary, thinking, content, text) are display text.
    const record = block as Record<string, unknown>

    const value = [record.summary, record.thinking, record.content, record.text].find(
      candidate => typeof candidate === 'string' && candidate.trim()
    )

    push(value)

    // A provider-native replay carrier keeps readable reasoning only in
    // nested blocks, never in its own opaque fields.
    walkNested(block)
  }

  return text.join('\n\n').trim()
}

/**
 * Known deliverable file extensions — mirrors the Python-side
 * `MEDIA_DELIVERY_EXTS` in `gateway/platforms/base.py` so the two surfaces
 * agree on which `MEDIA:` paths are valid. Used to anchor the end of an
 * unquoted path that may contain interior spaces (#96657).
 */
const MEDIA_DELIVERY_EXTS = [
  'png',
  'jpg',
  'jpeg',
  'gif',
  'webp',
  'bmp',
  'tiff',
  'svg',
  'mp4',
  'mov',
  'avi',
  'mkv',
  'webm',
  '3gp',
  'mp3',
  'm2a',
  'wav',
  'ogg',
  'opus',
  'm4a',
  'flac',
  'pdf',
  'docx',
  'doc',
  'odt',
  'rtf',
  'txt',
  'md',
  'epub',
  'xlsx',
  'xls',
  'ods',
  'csv',
  'tsv',
  'json',
  'xml',
  'yaml',
  'yml',
  'kmz',
  'kml',
  'geojson',
  'gpx',
  'pptx',
  'ppt',
  'odp',
  'key',
  'zip',
  'tar',
  'gz',
  'tgz',
  'bz2',
  'xz',
  '7z',
  'rar',
  'apk',
  'ipa',
  'html',
  'htm'
] as const

// Sort longest-first so the alternation never matches a shorter ext as a
// prefix of a longer one (e.g. `tar` before a hypothetical `tar.gz`).
const _MEDIA_EXT_ALTERNATION = [...MEDIA_DELIVERY_EXTS].sort((a, b) => b.length - a.length).join('|')

/**
 * Unquoted path branch: starts with a path anchor (`~/`, `/`, `X:\` or `X:/`),
 * allows interior whitespace, and anchors the end on a known deliverable
 * extension. Matches the Python-side `MEDIA_TAG_CLEANUP_RE` behavior where
 * `(?:[^\S\n]+\S+?)*?\.(?:EXT)` permits spaces inside filenames (#96657).
 */
const _MEDIA_PATH_ANCHORED = `(?:~/|/|[A-Za-z]:[/\\\\])\\S+?(?:[^\\S\\n]+\\S+?)*?\\.(?:${_MEDIA_EXT_ALTERNATION})(?=[\\s\`"'*_,;:)\\]}]|MEDIA:|$)`

// Bare-word fallback for paths the anchored branch misses (relative paths,
// unknown extensions). Stop before backtick and double-quote so an inline-code
// closer is not swallowed. Apostrophes stay legal inside the path.
const _MEDIA_PATH_BARE = '[^\\s`"]+'

// Sentence punctuation that can trail a bare capture when the tag sits in
// prose (`open MEDIA:/tmp/a.pdf.`).
const _MEDIA_TRAILING_PUNCTUATION = '.,;:!?'

/**
 * Whether a capture can name a real deliverable: a path separator, or a dot
 * with file content after it (an extension or a dotfile — `report.md`,
 * `.env`, `../a.png`). Anything else (`...`, a lone quote, a bare English
 * word) renders as prose: a dead `#media:` link for a non-path is worse
 * than no link (#84361).
 */
function isPlausibleMediaPath(value: string): boolean {
  return value.includes('/') || value.includes('\\') || /\.[^.]/.test(value)
}

const MEDIA_LINE_RE = new RegExp(
  `(^|\\n)[\\t ]*[\`"']?MEDIA:\\s*(?<line>\`[^\`\n]+\`|"[^"\n]+"|'[^'\n]+'|${_MEDIA_PATH_ANCHORED}|${_MEDIA_PATH_BARE})[\`"']?[\\t ]*(\\n|$)`,
  'g'
)

const MEDIA_TAG_RE = new RegExp(
  `[\`"']?MEDIA:\\s*(?<inline>\`[^\`\n]+\`|"[^"\n]+"|'[^'\n]+'|${_MEDIA_PATH_ANCHORED}|${_MEDIA_PATH_BARE})[\`"']?`,
  'g'
)

function unquoteMediaPath(value: string): string {
  const trimmed = value.trim()
  const quote = trimmed[0]

  if (quote && quote === trimmed.at(-1) && ['"', "'", '`'].includes(quote)) {
    return trimmed.slice(1, -1)
  }

  // A trailing backtick or double-quote left in the value is formatting residue,
  // not part of the path. Apostrophes are not residue (`john's.md`).
  const last = trimmed.at(-1)

  return last === '`' || last === '"' ? trimmed.slice(0, -1) : trimmed
}

/**
 * Split a bare (unquoted) capture into its path and the sentence punctuation
 * that trailed it in prose: `open MEDIA:/tmp/a.pdf.` captures `/tmp/a.pdf.` —
 * the period belongs to the sentence, not the path. Punctuation is only
 * prose while what precedes it still names a path, so an ellipsis
 * (`MEDIA:...`) is never split into a degenerate capture. Quoted captures are
 * exempt (quotes are the documented escape hatch for odd names:
 * `MEDIA:'/tmp/stop!.md'` keeps its `!`).
 */
function splitTrailingPunctuation(value: string): { path: string; punctuation: string } {
  let end = value.length

  while (end > 0 && _MEDIA_TRAILING_PUNCTUATION.includes(value[end - 1] ?? '')) {
    if (!isPlausibleMediaPath(value.slice(0, end - 1))) {
      break
    }

    end -= 1
  }

  return { path: value.slice(0, end), punctuation: value.slice(end) }
}

function mediaLink(value: string): string | null {
  const raw = value.trim()
  const quote = raw[0]
  const quoted = quote && quote === raw.at(-1) && ['"', "'", '`'].includes(quote)

  // Quoted captures are the escape hatch for odd names — punctuation inside
  // the quotes is part of the path, so only a BARE capture is split.
  const { path, punctuation } = quoted
    ? { path: unquoteMediaPath(raw), punctuation: '' }
    : splitTrailingPunctuation(unquoteMediaPath(raw))

  return isPlausibleMediaPath(path) ? `[${mediaDisplayLabel(path)}](${mediaMarkdownHref(path)})${punctuation}` : null
}

export function renderMediaTags(text: string): string {
  return text
    .replace(MEDIA_LINE_RE, (match, lead: string, value: string, trailer: string) => {
      const link = mediaLink(value)

      return link ? `${lead}${link}${trailer}` : match
    })
    .replace(MEDIA_TAG_RE, (match, value: string) => mediaLink(value) ?? match)
}

/** Raw `MEDIA:` values in `text`, quotes intact — the one parser Artifacts and chat share.
 *  Bare captures shed trailing sentence punctuation (same rule as
 *  {@link renderMediaTags}); degenerate non-path captures are dropped. */
export function mediaTagValues(text: string): string[] {
  return [...text.matchAll(MEDIA_TAG_RE)]
    .map(match => match[1] ?? '')
    .flatMap(value => {
      const { path, punctuation } = splitTrailingPunctuation(unquoteMediaPath(value))

      if (!isPlausibleMediaPath(path)) {
        return []
      }

      // Bare captures shed the prose punctuation (it trails the raw value
      // too); quoted captures keep every character inside their quotes.
      return [punctuation ? value.slice(0, -punctuation.length || undefined) : value]
    })
}

export function assistantTextPart(text: string, timestamp?: number): ChatMessagePart {
  return textPart(renderMediaTags(text), timestamp)
}

export function partsText(parts: ChatMessagePart[]): string {
  return parts
    .filter((part): part is Extract<ChatMessagePart, { type: 'text' }> => part.type === 'text')
    .map(part => part.text)
    .join('')
}

export function chatMessageText(message: ChatMessage): string {
  return partsText(message.parts)
}

export interface UnspokenTurnSpeech {
  /** First unspoken assistant bubble — stable for the turn, the live speech session binds to it. */
  id: string
  /** Whether the newest assistant bubble is still streaming. */
  pending: boolean
  /** All unspoken assistant text in message order, bubbles joined on a blank line. */
  text: string
}

/** Index of the nth visible assistant bubble (the anchor's durable "slot"), or -1. */
function indexOfAssistantOrdinal(messages: ChatMessage[], ordinal: number): number {
  let seen = -1

  for (const [index, message] of messages.entries()) {
    if (message.role !== 'assistant' || message.hidden) {
      continue
    }

    seen += 1

    if (seen === ordinal) {
      return index
    }
  }

  return -1
}

/**
 * Collect every unspoken assistant bubble after the spoken anchor, in order.
 *
 * A turn with tool calls produces several assistant bubbles — narration
 * ("Let me check…") sealed as interims, then the final answer as a fresh
 * bubble. Voice conversation speaks a turn through ONE live session bound to
 * one response id, so it needs all of that text as a single growing string;
 * selecting only one bubble silently drops everything after it. The blank-line
 * join is a sentence boundary for the server's cutter, so a sealed bubble's
 * tail is flushed as soon as the next bubble starts.
 *
 * If the anchor cannot be located — its row id was rewritten (stream → durable,
 * a re-hydration reassigned it) and its ordinal no longer lands on a visible
 * assistant bubble — say NOTHING. Re-reading a turn the session already spoke is
 * the worse failure (the text is on screen either way); the current-turn bound
 * below stays reserved for a session that has genuinely spoken nothing yet.
 * The row id alone is not a usable boundary: it is reassigned by every rewrite,
 * which is why the anchor carries the assistant ordinal (`spoken-reply.ts`) and
 * why this resolves by id FIRST and by that slot second. Id-only resolution is
 * how a spoken turn got read a second time, minutes after it had been read once.
 *
 * With a null anchor, bound to the current turn (assistant bubbles after the last
 * user message). Hidden user rows count: a widget intent (`display_kind: hidden`)
 * is a real turn for the agent even though no bubble renders. A slice with no
 * user row (mid-turn interims only) still collects those assistants.
 */
export function collectUnspokenTurnSpeech(
  messages: ChatMessage[],
  lastSpoken: SpokenReplyAnchor | null
): UnspokenTurnSpeech | null {
  let spokenIndex = -1

  if (lastSpoken) {
    spokenIndex = messages.findLastIndex(m => m.id === lastSpoken.id)

    if (spokenIndex < 0) {
      spokenIndex = indexOfAssistantOrdinal(messages, lastSpoken.ordinal)
    }

    if (spokenIndex < 0) {
      return null
    }
  } else {
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
 * paragraph twice and everything in it renders twice, most visibly ::preview
 * frames. Only that shape folds across rows: the bubble's final text (no tool
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

  const streamedText = normalizeWs(partsText(parts))

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

    const earlierText = partsText(earlier)

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
  const { index, parts: next } = appendStreamPart(parts, 'text', delta, timestamp)
  const part = next[index]

  if (part?.type !== 'text') {
    return next
  }

  // Re-render from the raw stream, never from the previous render: an unquoted
  // spaced path (`MEDIA:/tmp/AI Brain/report.pdf`) split across deltas would
  // otherwise settle on a card for `/tmp/AI` and keep the rest as prose (#96657).
  const previous = parts[index]
  const source = `${previous?.type === 'text' ? (previous.mediaSource ?? previous.text) : ''}${delta}`

  if (!source.includes('MEDIA:')) {
    return next
  }

  const rendered = renderMediaTags(source)

  next[index] = rendered === source ? { ...part, text: source } : { ...part, mediaSource: source, text: rendered }

  return next
}
