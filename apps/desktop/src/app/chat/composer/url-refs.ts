/**
 * Bare-link recognition for the composer. A link the user pastes or types is the
 * same thing the "+ → Add URL" dialog inserts, so it becomes an `@url:`
 * directive: a chip that truncates instead of a wall of URL text, and a
 * reference the gateway resolves.
 */
import type { KeyboardEvent } from 'react'

import { quoteRefValue, REF_RE, refChipElement, replaceBeforeCaret } from './rich-editor'
import { textBeforeCaret } from './text-utils'

// An explicit scheme only — `example.com` bare is too easy to hit by accident
// (a filename, a version, a sentence). Brackets and quotes fence a URL in prose;
// parens don't, so they stay in and an unbalanced tail is trimmed below.
const URL_RE = /https?:\/\/[^\s<>[\]{}"'`]+/gi
const TYPED_URL_RE = /(?:^|\s)(https?:\/\/[^\s<>[\]{}"'`]+)$/i

interface TextRange {
  end: number
  start: number
}

const containsIndex = (ranges: TextRange[], index: number) =>
  ranges.some(range => index >= range.start && index < range.end)

function isEscaped(text: string, index: number) {
  let backslashes = 0

  for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) {
    backslashes += 1
  }

  return backslashes % 2 === 1
}

/** Markdown fenced code blocks, including an unfinished block while composing. */
function fencedCodeRanges(text: string) {
  const ranges: TextRange[] = []
  let opening: { marker: string; start: number } | undefined

  for (const match of text.matchAll(/^[ \t]{0,3}(`{3,}|~{3,})([^\r\n]*)(?:\r?\n|$)/gm)) {
    const marker = match[1]
    const tail = match[2]

    if (!opening) {
      // Backticks in an opening backtick fence's info string are invalid
      // Markdown, so do not turn the rest of the draft into protected code.
      if (marker[0] === '`' && tail.includes('`')) {
        continue
      }

      opening = { marker, start: match.index ?? 0 }

      continue
    }

    const closesFence = marker[0] === opening.marker[0] && marker.length >= opening.marker.length && tail.trim() === ''

    if (closesFence) {
      ranges.push({ end: (match.index ?? 0) + match[0].length, start: opening.start })
      opening = undefined
    }
  }

  if (opening) {
    ranges.push({ end: text.length, start: opening.start })
  }

  return ranges
}

/** Markdown inline code spans outside fences, including an unfinished span. */
function inlineCodeRanges(text: string, fenced: TextRange[]) {
  const ranges: TextRange[] = []

  const markers = Array.from(text.matchAll(/`+/g)).filter(marker => {
    const index = marker.index ?? 0

    return !containsIndex(fenced, index) && !isEscaped(text, index)
  })

  let markerIndex = 0

  while (markerIndex < markers.length) {
    const opening = markers[markerIndex]

    const closingIndex = markers.findIndex(
      (candidate, index) => index > markerIndex && candidate[0].length === opening[0].length
    )

    if (closingIndex === -1) {
      ranges.push({ end: text.length, start: opening.index ?? 0 })

      break
    }

    const closing = markers[closingIndex]

    ranges.push({
      end: (closing.index ?? 0) + closing[0].length,
      start: opening.index ?? 0
    })
    markerIndex = closingIndex + 1
  }

  return ranges
}

function markdownCodeRanges(text: string) {
  const fenced = fencedCodeRanges(text)

  return [...fenced, ...inlineCodeRanges(text, fenced)]
}

/** A URL at the end of a sentence carries the punctuation that ended it. */
function splitUrlTail(raw: string) {
  let url = raw.replace(/[,.;:!?]+$/, '')

  while (url.endsWith(')') && url.split(')').length > url.split('(').length) {
    url = url.slice(0, -1)
  }

  return { trailing: raw.slice(url.length), url }
}

/** A URL needs a host past the scheme to be worth chipping. */
const hasHost = (url: string) => /^https?:\/\/[^/\s]/i.test(url)

/** Rewrite bare links in `text` as `@url:` directives, leaving links that are
 *  already part of a directive alone. Returns `text` unchanged when there are
 *  none. */
export function linkifyUrls(text: string) {
  REF_RE.lastIndex = 0

  const protectedRanges = Array.from(text.matchAll(REF_RE)).map(match => {
    const start = match.index ?? 0

    return { end: start + match[0].length, start }
  })

  protectedRanges.push(...markdownCodeRanges(text))

  let out = ''
  let cursor = 0

  for (const match of text.matchAll(URL_RE)) {
    const start = match.index ?? 0
    const { url } = splitUrlTail(match[0])

    if (!hasHost(url) || containsIndex(protectedRanges, start)) {
      continue
    }

    out += `${text.slice(cursor, start)}@url:${quoteRefValue(url)}`
    cursor = start + url.length
  }

  return out + text.slice(cursor)
}

/** A plain space finishing a typed link commits it as a chip (followed by
 *  whatever punctuation ended it, then the space). Returns whether it ran, so a
 *  keydown handler can fall through on anything else. */
export function chipTypedUrlOnSpace(event: KeyboardEvent<HTMLDivElement>) {
  if (event.key !== ' ' || event.metaKey || event.ctrlKey || event.altKey) {
    return false
  }

  const editor = event.currentTarget

  // Runs on every space, so bail on the cheap native read before paying for the
  // caret range walk (same guard shape as the trigger detector).
  if (!editor.textContent?.includes('://')) {
    return false
  }

  const before = textBeforeCaret(editor)

  if (!before) {
    return false
  }

  const match = TYPED_URL_RE.exec(before)
  const token = match?.[1]

  if (!token) {
    return false
  }

  const tokenStart = before.length - token.length

  if (containsIndex(markdownCodeRanges(before), tokenStart)) {
    return false
  }

  const { trailing, url } = splitUrlTail(token)

  if (!hasHost(url)) {
    return false
  }

  const fragment = document.createDocumentFragment()

  fragment.append(refChipElement('url', quoteRefValue(url)))

  if (trailing) {
    fragment.append(document.createTextNode(trailing))
  }

  fragment.append(document.createTextNode(' '))

  return replaceBeforeCaret(editor, token.length, fragment)
}
