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
// Anchored, non-global twin for the whole-payload check: `exec` on the shared
// global `URL_RE` would leave its `lastIndex` past the match, and `matchAll`
// inherits that offset, so the `linkifyUrls` call later in the same paste
// handler would start scanning after the link and chip nothing.
const EXACT_URL_RE = /^https?:\/\/[^\s<>[\]{}"'`]+$/i
const TYPED_URL_RE = /(?:^|\s)(https?:\/\/[^\s<>[\]{}"'`]+)$/i

/** A half-open `[start, end)` span of the draft text. */
interface TextRange {
  end: number
  start: number
}

const containsIndex = (ranges: TextRange[], index: number) =>
  ranges.some(range => index >= range.start && index < range.end)

/** A backtick run preceded by an odd number of backslashes is escaped prose, not
 *  a code delimiter (`\`` reads literally). */
function isEscaped(text: string, index: number) {
  let backslashes = 0

  for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) {
    backslashes += 1
  }

  return backslashes % 2 === 1
}

/** Markdown fenced code blocks (``` or ~~~), including an unfinished block while
 *  the user is still composing one. */
function fencedCodeRanges(text: string) {
  const ranges: TextRange[] = []
  let opening: { marker: string; start: number } | undefined

  for (const match of text.matchAll(/^[ \t]{0,3}(`{3,}|~{3,})([^\r\n]*)(?:\r?\n|$)/gm)) {
    const marker = match[1]
    const tail = match[2]

    if (!opening) {
      // Backticks in a backtick fence's info string are invalid Markdown, so a
      // stray run must not turn the rest of the draft into protected code.
      if (marker[0] === '`' && tail.includes('`')) {
        continue
      }

      opening = { marker, start: match.index ?? 0 }

      continue
    }

    // A closing fence is the same character, at least as long, with nothing but
    // whitespace after it.
    if (marker[0] === opening.marker[0] && marker.length >= opening.marker.length && tail.trim() === '') {
      ranges.push({ end: (match.index ?? 0) + match[0].length, start: opening.start })
      opening = undefined
    }
  }

  if (opening) {
    ranges.push({ end: text.length, start: opening.start })
  }

  return ranges
}

/** Markdown inline code spans outside fences, including an unfinished span while
 *  the user is still composing one. Runs pair up by matching backtick length, so
 *  `` `a ` b` `` stays one span. */
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

    ranges.push({ end: (closing.index ?? 0) + closing[0].length, start: opening.index ?? 0 })
    markerIndex = closingIndex + 1
  }

  return ranges
}

/** The spans of `text` Markdown renders as code — fenced blocks plus inline
 *  spans. A URL inside one is verbatim payload (a stack trace, a log line, a
 *  command), not prose to chip, so the recognizer must leave it alone. */
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

/** A URL needs a host past the scheme to be worth treating as a URL reference. */
export const hasHttpUrlHost = (url: string) => /^https?:\/\/[^/\s]/i.test(splitUrlTail(url).url)

/** Rewrite bare links in `text` as `@url:` directives, leaving links that are
 *  already part of a directive alone. Returns `text` unchanged when there are
 *  none. */
export function linkifyUrls(text: string) {
  REF_RE.lastIndex = 0

  // URLs inside an existing `@url:` directive or a Markdown code span are not
  // prose to chip — the directive is already a reference, and code is verbatim
  // payload the user pasted (a stack trace, a command, a log line).
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

    if (!hasHttpUrlHost(url) || containsIndex(protectedRanges, start)) {
      continue
    }

    out += `${text.slice(cursor, start)}@url:${quoteRefValue(url)}`
    cursor = start + url.length
  }

  return out + text.slice(cursor)
}

/** The href to apply when a clipboard payload is exactly ONE supported link —
 *  a bare or `<…>`-wrapped `http(s)` URL with a host and nothing else. Null for
 *  anything that isn't a lone link (prose, multiple links, trailing text), so
 *  callers fall through to the normal paste pipeline. Ported from
 *  block/buzz#6684's `resolveExactLinkPaste`. */
export function resolveExactLinkPaste(raw: string): string | null {
  const text = raw.trim()
  const unwrapped = text.startsWith('<') && text.endsWith('>') && text.length > 2 ? text.slice(1, -1).trim() : text

  if (!EXACT_URL_RE.test(unwrapped)) {
    return null
  }

  const { trailing, url } = splitUrlTail(unwrapped)

  // Trailing sentence punctuation means the user copied prose, not a link.
  if (trailing || !hasHttpUrlHost(url)) {
    return null
  }

  return url
}

/** The selected composer text a link paste should hyperlink, or null when the
 *  selection can't take a link mark: collapsed, outside `editor`, spanning
 *  chips or line breaks, or whitespace-only. */
export function selectionLinkLabel(editor: HTMLElement): string | null {
  const selection = window.getSelection()

  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
    return null
  }

  const range = selection.getRangeAt(0)

  if (!editor.contains(range.commonAncestorContainer)) {
    return null
  }

  const probe = document.createElement('div')

  probe.append(range.cloneContents())

  // A chip inside the selection is a directive, not prose — linking over it
  // would destroy the reference. Multi-line selections don't read as a label.
  if (probe.querySelector('[data-ref-text], br')) {
    return null
  }

  const label = probe.textContent?.replace(/\s+/g, ' ').trim() ?? ''

  return label || null
}

/** Markdown link for a paste-over-selection: the label the user selected, the
 *  URL they pasted. Square brackets in the label are escaped so the link
 *  survives markdown parsing downstream. */
export function markdownLinkFor(label: string, url: string): string {
  return `[${label.replace(/([[\]])/g, '\\$1')}](${url})`
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

  // A link typed inside a code block or span is verbatim payload, not prose —
  // chipping it would rewrite code the user is authoring.
  if (containsIndex(markdownCodeRanges(before), before.length - token.length)) {
    return false
  }

  const { trailing, url } = splitUrlTail(token)

  if (!hasHttpUrlHost(url)) {
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
