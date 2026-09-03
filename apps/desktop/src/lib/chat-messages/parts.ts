import { mediaDisplayLabel, mediaMarkdownHref } from '@/lib/media'

import type { ChatMessage, ChatMessagePart } from './types'

export function textPart(text: string, timestamp?: number): ChatMessagePart {
  return { type: 'text', text, ...(timestamp !== undefined ? { timestamp } : {}) }
}

export function reasoningPart(text: string, timestamp?: number): ChatMessagePart {
  return { type: 'reasoning', text, ...(timestamp !== undefined ? { timestamp } : {}) }
}

const MEDIA_LINE_RE =
  /(^|\n)[\t ]*[`"']?MEDIA:(?![\t ]*(?:\r?\n|$))[\t ]*(?<line>`[^`\r\n]+`|"[^"\r\n]+"|'[^'\r\n]+'|[^\r\n]*?\S)[`"']?[\t ]*(?=\r?\n|$)/g

const MEDIA_TAG_RE = /[`"']?MEDIA:[\t ]*(?<inline>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)[`"']?/g

function unquoteMediaPath(value: string): string {
  const trimmed = value.trim()
  const quote = trimmed[0]

  return quote && quote === trimmed.at(-1) && ['"', "'", '`'].includes(quote) ? trimmed.slice(1, -1) : trimmed
}

function mediaLink(value: string, onPath?: (path: string) => void): string {
  const path = unquoteMediaPath(value)

  onPath?.(path)

  return `[${mediaDisplayLabel(path)}](${mediaMarkdownHref(path)})`
}

function isExplicitlyQuotedMediaPath(value: string): boolean {
  const trimmed = value.trim()
  const quote = trimmed[0]

  return Boolean(quote && quote === trimmed.at(-1) && ['"', "'", '`'].includes(quote))
}

function isClosedQuotedMediaDirective(match: string, lead: string): boolean {
  return isExplicitlyQuotedMediaPath(match.slice(lead.length))
}

function wholeDirectiveOpeningQuote(match: string, lead: string): string | undefined {
  const directive = match.slice(lead.length).trimStart()
  const quote = directive[0]

  return quote && ['"', "'", '`'].includes(quote) && directive.slice(1).startsWith('MEDIA:') ? quote : undefined
}

function isPathShapedMediaValue(value: string): boolean {
  return (
    /^(?:file:|https?:|data:|\/|~[\\/]|\.{1,2}[\\/]|[a-z]:[\\/]|\\\\)/i.test(value) ||
    /\.[a-z0-9]{1,16}(?:[?#].*)?$/i.test(value)
  )
}

type StandaloneMediaIntent = 'inline' | 'path' | 'prose'

type StandaloneMediaValue = { intent: 'inline' | 'prose' } | { intent: 'path'; path: string; tail: string }

const MEDIA_EXTENSION_RE = /\.[a-z0-9]{1,16}(?:[?#].*)?$/i

/**
 * Classify a line-leading MEDIA value before replacing it. A quoted value or
 * `/tmp/the final report.pdf` is one path; `the report is ready` is prose;
 * `/tmp/report.pdf is ready` splits at the extension so the suffix remains
 * prose instead of becoming part of the filename.
 */
function parseStandaloneMediaValue(value: string): StandaloneMediaValue {
  const trimmed = value.trim()

  if (!trimmed) {
    return { intent: 'prose' }
  }

  const quote = trimmed[0]

  if (['"', "'", '`'].includes(quote)) {
    const closingQuote = trimmed.indexOf(quote, 1)

    if (closingQuote > 0) {
      return {
        intent: 'path',
        path: trimmed.slice(0, closingQuote + 1),
        tail: trimmed.slice(closingQuote + 1)
      }
    }
  }

  if (!/\s/.test(trimmed)) {
    return isPathShapedMediaValue(trimmed) ? { intent: 'path', path: trimmed, tail: '' } : { intent: 'prose' }
  }

  if (/\bMEDIA:/.test(trimmed)) {
    return { intent: 'inline' }
  }

  const firstToken = trimmed.match(/^\S+/)?.[0]

  if (firstToken && /^(?:https?:|data:)/i.test(firstToken)) {
    return { intent: 'path', path: firstToken, tail: trimmed.slice(firstToken.length) }
  }

  if (MEDIA_EXTENSION_RE.test(trimmed)) {
    return { intent: 'path', path: trimmed, tail: '' }
  }

  if (isPathShapedMediaValue(trimmed)) {
    const tokens = [...trimmed.matchAll(/\S+/g)]

    for (let index = tokens.length - 1; index >= 0; index -= 1) {
      const token = tokens[index]

      if (!MEDIA_EXTENSION_RE.test(token[0])) {
        continue
      }

      const end = (token.index ?? 0) + token[0].length

      return { intent: 'path', path: trimmed.slice(0, end), tail: trimmed.slice(end) }
    }

    return /^(?:file:|~[\\/]|\.{1,2}[\\/]|[a-z]:[\\/]|\\\\|\/[^/\s]+[\\/])/i.test(trimmed)
      ? { intent: 'path', path: trimmed, tail: '' }
      : { intent: 'prose' }
  }

  if (firstToken && MEDIA_EXTENSION_RE.test(firstToken)) {
    return { intent: 'path', path: firstToken, tail: trimmed.slice(firstToken.length) }
  }

  return { intent: 'prose' }
}

function standaloneMediaIntent(value: string): StandaloneMediaIntent {
  return parseStandaloneMediaValue(value).intent
}

interface ConsecutiveMediaMarker {
  end: number
  index: number
  mediaIndex: number
}

function isHorizontalWhitespace(value: string | undefined): boolean {
  return value === ' ' || value === '\t'
}

function trimHorizontalWhitespaceBeforeNewlines(value: string): string {
  let rendered = ''
  let lineStart = 0

  while (lineStart < value.length) {
    const newline = value.indexOf('\n', lineStart)

    if (newline < 0) {
      rendered += value.slice(lineStart)

      break
    }

    const carriageReturn = newline > lineStart && value[newline - 1] === '\r'
    let contentEnd = carriageReturn ? newline - 1 : newline

    while (contentEnd > lineStart && isHorizontalWhitespace(value[contentEnd - 1])) {
      contentEnd -= 1
    }

    rendered += `${value.slice(lineStart, contentEnd)}${carriageReturn ? '\r' : ''}\n`
    lineStart = newline + 1
  }

  return rendered
}

function consecutiveMediaMarkers(value: string): ConsecutiveMediaMarker[] {
  const markers: ConsecutiveMediaMarker[] = []
  let searchFrom = 0

  while (searchFrom < value.length) {
    const mediaIndex = value.indexOf('MEDIA:', searchFrom)

    if (mediaIndex < 0) {
      break
    }

    const quoted = ['`', '"', "'"].includes(value[mediaIndex - 1] || '')
    const prefixEnd = quoted ? mediaIndex - 1 : mediaIndex

    if (!isHorizontalWhitespace(value[prefixEnd - 1])) {
      searchFrom = mediaIndex + 'MEDIA:'.length

      continue
    }

    let index = prefixEnd - 1

    while (index > 0 && isHorizontalWhitespace(value[index - 1])) {
      index -= 1
    }

    let end = mediaIndex + 'MEDIA:'.length

    while (end < value.length && isHorizontalWhitespace(value[end])) {
      end += 1
    }

    markers.push({ end, index, mediaIndex })
    searchFrom = end
  }

  return markers
}

function isOpenConsecutiveMediaMarker(value: string, markers: ConsecutiveMediaMarker[]): boolean {
  const marker = markers.at(-1)

  return marker ? /^[\t ]*$/.test(value.slice(marker.mediaIndex + 'MEDIA:'.length)) : false
}

function renderConsecutiveMediaValues(
  value: string,
  onPath: (path: string) => void,
  markers = consecutiveMediaMarkers(value)
): string | undefined {
  if (!markers.length) {
    return undefined
  }

  const values: string[] = []
  let start = 0

  for (const marker of markers) {
    values.push(value.slice(start, marker.index).trim())
    start = marker.end
  }

  values.push(value.slice(start).trim())
  const parsed = values.map(parseStandaloneMediaValue)

  if (parsed.some(entry => entry.intent !== 'path')) {
    return undefined
  }

  return parsed
    .map(entry => {
      if (entry.intent !== 'path') {
        return ''
      }

      return `${mediaLink(entry.path, onPath)}${entry.tail}`
    })
    .join(' ')
}

function renderMediaSequencesInText(text: string, onPath: (path: string) => void, onOpenSequence: () => void): string {
  let rendered = ''
  let lineStart = 0

  while (lineStart < text.length) {
    const newline = text.indexOf('\n', lineStart)
    const lineEnd = newline === -1 ? text.length : newline
    const contentEnd = lineEnd > lineStart && text[lineEnd - 1] === '\r' ? lineEnd - 1 : lineEnd
    const line = text.slice(lineStart, contentEnd)
    const marker = line.indexOf('MEDIA:')
    let content = line

    if (marker >= 0) {
      const value = line.slice(marker + 'MEDIA:'.length)
      const markers = consecutiveMediaMarkers(value)
      const sequence = renderConsecutiveMediaValues(value, onPath, markers)

      if (sequence !== undefined) {
        content = `${line.slice(0, marker)}${sequence}`

        if (newline === -1) {
          onOpenSequence()
        }
      } else if (newline === -1 && isOpenConsecutiveMediaMarker(value, markers)) {
        onOpenSequence()
      }
    }

    rendered += content

    if (newline === -1) {
      break
    }

    rendered += text.slice(contentEnd, newline + 1)
    lineStart = newline + 1
  }

  return rendered
}

function mediaSequenceIntentsByLine(source: string): Map<number, StandaloneMediaIntent> {
  const intents = new Map<number, StandaloneMediaIntent>()
  let lineStart = 0

  while (lineStart <= source.length) {
    const lineEnd = source.indexOf('\n', lineStart)
    const line = source.slice(lineStart, lineEnd === -1 ? source.length : lineEnd)
    const marker = line.indexOf('MEDIA:')

    if (marker >= 0) {
      const value = line.slice(marker + 'MEDIA:'.length)
      const markers = consecutiveMediaMarkers(value)

      if (markers.length) {
        const intent = renderConsecutiveMediaValues(value, () => {}, markers) === undefined ? 'prose' : 'path'

        intents.set(lineStart, intent)
      }
    }

    if (lineEnd === -1) {
      break
    }

    lineStart = lineEnd + 1
  }

  return intents
}

function lineLeadingMediaIntent(source: string, offset: number): StandaloneMediaIntent | undefined {
  const lineStart = source.lastIndexOf('\n', Math.max(0, offset - 1)) + 1

  if (!/^[\t ]*$/.test(source.slice(lineStart, offset))) {
    return undefined
  }

  const lineEnd = source.indexOf('\n', offset)
  const line = source.slice(offset, lineEnd === -1 ? source.length : lineEnd)
  const marker = line.indexOf('MEDIA:')

  return marker === -1 ? undefined : standaloneMediaIntent(line.slice(marker + 'MEDIA:'.length))
}

function renderMediaText(text: string): { paths: string[]; provisional: boolean; text: string } {
  const paths: string[] = []
  let provisional = false

  let rendered = text.replace(MEDIA_LINE_RE, (match, lead: string, value: string, offset: number, source: string) => {
    if (/[\t ]+[`"']?MEDIA:[\t ]*$/.test(value) && offset + match.length === source.length) {
      provisional = true

      return match
    }

    const sequence = renderConsecutiveMediaValues(value, path => paths.push(path))

    if (sequence !== undefined) {
      provisional ||= offset + match.length === source.length

      return `${lead}${sequence}`
    }

    const wholeDirectiveQuote = wholeDirectiveOpeningQuote(match, lead)
    const closingWholeDirectiveQuote = wholeDirectiveQuote ? value.indexOf(wholeDirectiveQuote) : -1
    let parsed = parseStandaloneMediaValue(value)
    let wholeDirectiveClosed = isClosedQuotedMediaDirective(match, lead)
    const trailingWhitespace = /[\t ]+$/.test(match)

    if (closingWholeDirectiveQuote >= 0) {
      parsed = {
        intent: 'path',
        path: value.slice(0, closingWholeDirectiveQuote),
        tail: value.slice(closingWholeDirectiveQuote + 1)
      }
      wholeDirectiveClosed = true
    }

    if (parsed.intent !== 'path') {
      return match
    }

    provisional ||=
      !isExplicitlyQuotedMediaPath(parsed.path) &&
      (!wholeDirectiveClosed || trailingWhitespace) &&
      offset + match.length === source.length

    return `${lead}${mediaLink(parsed.path, path => paths.push(path))}${parsed.tail}`
  })

  rendered = renderMediaSequencesInText(
    rendered,
    path => paths.push(path),
    () => {
      provisional = true
    }
  )

  const sequenceIntents = mediaSequenceIntentsByLine(rendered)
  let inlineLineStart = 0
  let inlineLineEnd = rendered.indexOf('\n')

  rendered = rendered.replace(MEDIA_TAG_RE, (match, value: string, offset: number, source: string) => {
    while (inlineLineEnd >= 0 && offset > inlineLineEnd) {
      inlineLineStart = inlineLineEnd + 1
      inlineLineEnd = rendered.indexOf('\n', inlineLineStart)
    }

    const sequenceIntent = sequenceIntents.get(inlineLineStart)

    if (sequenceIntent === 'prose') {
      return match
    }

    const lineIntent = lineLeadingMediaIntent(source, offset)
    const pathShaped = isExplicitlyQuotedMediaPath(value) || isPathShapedMediaValue(unquoteMediaPath(value))

    if (lineIntent === 'prose' || (lineIntent === undefined && !pathShaped)) {
      return match
    }

    provisional ||= !isExplicitlyQuotedMediaPath(value) && source.indexOf('\n', offset) === -1

    return mediaLink(value, path => paths.push(path))
  })

  rendered = trimHorizontalWhitespaceBeforeNewlines(rendered).replace(/\n{3,}/g, '\n\n')

  return { paths, provisional, text: rendered }
}

export function renderMediaTags(text: string): string {
  return renderMediaText(text).text
}

export function mediaPathsFromText(text: string): string[] {
  return renderMediaText(text).paths
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
  parts = clearProvisionalMediaSources(parts)

  // Empty / whitespace-only completion is not authoritative — keep streamed
  // text, reasoning, and tool parts (#95514). Provisional MEDIA metadata is
  // still settled first so it cannot leak past the turn boundary.
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
  return parts.map(part => {
    const settled = clearProvisionalMediaSource(part)

    return settled.timestamp !== undefined && settled.completedAt === undefined
      ? ({ ...settled, completedAt } as ChatMessagePart)
      : settled
  })
}

function clearProvisionalMediaSource(part: ChatMessagePart): ChatMessagePart {
  if (part.provisionalMediaSource === undefined) {
    return part
  }

  const { provisionalMediaSource: _source, ...settled } = part

  return settled as ChatMessagePart
}

function clearProvisionalMediaSources(parts: ChatMessagePart[]): ChatMessagePart[] {
  let changed = false

  const settled = parts.map(part => {
    const next = clearProvisionalMediaSource(part)

    changed ||= next !== part

    return next
  })

  return changed ? settled : parts
}

function restoreOpenProvisionalMediaSource(parts: ChatMessagePart[]): ChatMessagePart[] {
  const tailIndex = parts.length - 1
  const tail = parts[tailIndex]

  if (tail?.type !== 'text' || tail.completedAt !== undefined || tail.provisionalMediaSource === undefined) {
    return parts
  }

  const next = [...parts]
  const settled = clearProvisionalMediaSource(tail)

  next[tailIndex] = { ...settled, text: tail.provisionalMediaSource } as ChatMessagePart

  return next
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

  if ((tail?.type === 'text' || tail?.type === 'reasoning') && tail.completedAt === undefined) {
    const settled = clearProvisionalMediaSource(tail)

    next[tailIndex] = {
      ...settled,
      ...(timestamp !== undefined ? { completedAt: timestamp } : {})
    } as ChatMessagePart
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
  const { index, parts: next } = appendStreamPart(restoreOpenProvisionalMediaSource(parts), 'text', delta, timestamp)
  const part = next[index]

  if (part?.type !== 'text') {
    return next
  }

  const mayContainMedia =
    delta.includes('MEDIA:') || delta.includes('DIA:') || delta.includes('EDIA:') || delta.includes('IA:')

  if (mayContainMedia || part.text.includes('MEDIA:')) {
    const rendered = renderMediaText(part.text)

    if (rendered.text !== part.text || rendered.provisional) {
      const settled = clearProvisionalMediaSource(part)

      next[index] = {
        ...settled,
        text: rendered.text,
        ...(rendered.provisional ? { provisionalMediaSource: part.text } : {})
      } as ChatMessagePart
    }
  }

  return next
}
